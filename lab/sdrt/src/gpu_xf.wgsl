// Transformation (op 0x04): a 2x2 matrix and an offset map output pixels to source
// coordinates. The CPU reference (ops.rs "transformation") picks one mip level, then
// samples it with nearest or the texture unit's 1/256 bilinear; auto-mip bilinear signed
// permutations instead resize the source per axis and permute its texels.
//
// The matrix maps output to source directly, column-major:
//   uv_src = (m0*dx + m2*dy, m1*dx + m3*dy) + 0.5 + offset
// with dx = (x + 0.5)/w - 0.5.
//
// ops.rs try_gpu_transform runs one node as a chain of passes of this kernel
// (gpu.rs dispatch_transform); config.header.x selects the pass:
//   0 sample   - the matrix/offset sampling of one level (image.rs "bilinear_or", "nearest")
//   1 permute  - the dihedral path's texel permutation (also the identity copy)
//   2 halve    - one mip level / per-axis halving (image.rs "mips", "downsample_axes")
//   3 upsample - one per-axis 2x bilinear upsample (image.rs "upsample_axes_fmt")
// The output pass format and colour flag (options.x/.y) quantize as the CPU stores the
// intermediate (halvings quantize to their format; upsamples are unquantized: f32, colour).
//
// config.header.w bits: 0 wraps x, 1 wraps y, 2 bilinear, 3 clamps the border (no matte),
// 8 / 9 halve or upsample along x / y. config.parameters[0] is the matrix, parameters[1].xy
// the offset, config.extra the matte colour, config.selectors the input descriptor (width,
// height, sampled storage format, colour). Storage is RGBA f32 exactly as the CPU stores it.
// Products feeding additions go through the prelude's barrier(), so every multiply and add
// rounds on its own as on the CPU.

// The input descriptor: width, height, storage format, colour flag.
fn source_desc() -> vec4<u32> {
    return config.selectors;
}

// Rust's f32::round: half away from zero (WGSL's round is ties-to-even).
fn round_away(v: f32) -> f32 {
    if v >= 0.0 { return floor(v + 0.5); }
    return -floor(-v + 0.5);
}

// i.rem_euclid(n) for n > 0. Never `%` on a negative i32: on wgpu 0.19 / NVIDIA 615 it gives the
// unsigned remainder of the bit pattern (-5 % 3 == 2), right only for powers of two.
fn wrap_index(i: i32, n: i32) -> i32 {
    if i >= 0 { return i % n; }
    // -(i + 1) >= 0 for every i < 0 (no overflow at i32::MIN)
    return n - 1 - i32(u32(-(i + 1)) % u32(n));
}

fn is_nan(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;
}

// Texel fetch for the bilinear path. A texel past a non-tiled edge is the matte colour,
// or the edge texel itself when the border clamps (the CPU's border=None case: an
// identity matrix with a manual level, and the upsamples). A tiled axis wraps.
fn fetch_or(x: i32, y: i32, wrap_x: bool, wrap_y: bool, clamp_border: bool, matte: vec4<f32>) -> vec4<f32> {
    let w = i32(source_desc().x);
    let h = i32(source_desc().y);
    var xx = x;
    var yy = y;
    if xx < 0 || xx >= w {
        if wrap_x { xx = wrap_index(xx, w); }
        else if clamp_border { xx = clamp(xx, 0, w - 1); }
        else { return matte; }
    }
    if yy < 0 || yy >= h {
        if wrap_y { yy = wrap_index(yy, h); }
        else if clamp_border { yy = clamp(yy, 0, h - 1); }
        else { return matte; }
    }
    return input_a[u32(yy * w + xx)];
}

// The CPU's bilinear sum ((a*w0 + b*w1) + c*w2) + d*w3, each product and sum rounded.
fn weighted(a: f32, b: f32, c: f32, d: f32, wt: vec4<f32>) -> f32 {
    var o = barrier(a * wt.x);
    o = o + barrier(b * wt.y);
    o = o + barrier(c * wt.z);
    o = o + barrier(d * wt.w);
    return o;
}

// A UNORM texel's 16-bit code, (v * 65535).round() as u32 as the CPU converts it.
fn code16(v: vec4<f32>) -> vec4<u32> {
    return vec4<u32>(
        u32(round_away(barrier(v.x * 65535.0))), u32(round_away(barrier(v.y * 65535.0))),
        u32(round_away(barrier(v.z * 65535.0))), u32(round_away(barrier(v.w * 65535.0))));
}

fn sample(u: f32, v: f32, bilinear: bool, wrap_x: bool, wrap_y: bool, clamp_border: bool, matte: vec4<f32>) -> vec4<f32> {
    let wi = i32(source_desc().x);
    let hi = i32(source_desc().y);
    let w = f32(wi);
    let h = f32(hi);
    if !bilinear {
        // The CPU's nearest path: a coordinate outside a non-tiled axis takes the matte;
        // otherwise image.rs "fetch" wraps tiled axes and clamps the others.
        let outside = (!wrap_x && !(u >= 0.0 && u <= 1.0)) || (!wrap_y && !(v >= 0.0 && v <= 1.0));
        if outside { return matte; }
        var xx = i32(floor(u * w));
        var yy = i32(floor(v * h));
        if wrap_x { xx = wrap_index(xx, wi); } else { xx = clamp(xx, 0, wi - 1); }
        if wrap_y { yy = wrap_index(yy, hi); } else { yy = clamp(yy, 0, hi - 1); }
        return input_a[u32(yy * wi + xx)];
    }
    let x = barrier(u * w) - 0.5;
    let y = barrier(v * h) - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    // The CPU rounds the fractions with "ties away from zero" (Rust's f32::round). Every
    // step is exact: x - x0 is representable and the scales are powers of two.
    let fx = round_away((x - x0) * 256.0) * (1.0 / 256.0);
    let fy = round_away((y - y0) * 256.0) * (1.0 / 256.0);
    // The cross weight rounds to 1/256 too, the others follow from it (all exact).
    let cross = round_away(fx * fy * 256.0) * (1.0 / 256.0);
    let wt = vec4<f32>(1.0 - fx - fy + cross, fx - cross, fy - cross, cross);
    let xi = i32(x0);
    let yi = i32(y0);
    let a = fetch_or(xi, yi, wrap_x, wrap_y, clamp_border, matte);
    let b = fetch_or(xi + 1, yi, wrap_x, wrap_y, clamp_border, matte);
    let c = fetch_or(xi, yi + 1, wrap_x, wrap_y, clamp_border, matte);
    let d = fetch_or(xi + 1, yi + 1, wrap_x, wrap_y, clamp_border, matte);
    if source_desc().z <= 1u {
        // UNORM sources interpolate in 16-bit integer space, ties up. The CPU truncates the
        // weights with (w * 256) as u32 (exact: they are multiples of 1/256).
        let wq = vec4<u32>(wt * 256.0);
        let sum = code16(a) * wq.x + code16(b) * wq.y + code16(c) * wq.z + code16(d) * wq.w;
        let code = (sum + vec4<u32>(128u)) >> vec4<u32>(8u);
        // code / 65535 correctly rounded, as the CPU's division (not the GPU's approximate one).
        return vec4<f32>(
            unorm_float(code.x, 16u), unorm_float(code.y, 16u),
            unorm_float(code.z, 16u), unorm_float(code.w, 16u));
    }
    return vec4<f32>(
        weighted(a.x, b.x, c.x, d.x, wt), weighted(a.y, b.y, c.y, d.y, wt),
        weighted(a.z, b.z, c.z, d.z, wt), weighted(a.w, b.w, c.w, d.w, wt));
}

// The CPU's dihedral path after its resize (auto mip, bilinear, signed-permutation matrix,
// zero offset): output texel (x, y) is input texel (sx, sy), no arithmetic. For [0,±1,±1,0]
// the input is h x w (the output transposed).
fn permute_pixel(k: u32) -> vec4<f32> {
    let w = config.header.z;
    let h = config.header.y / w;
    let x = k % w;
    let y = k / w;
    let m = config.parameters[0];
    var sx: u32;
    var sy: u32;
    if m.x != 0.0 {
        sx = select(w - 1u - x, x, m.x > 0.0);
        sy = select(h - 1u - y, y, m.w > 0.0);
    } else {
        sx = select(h - 1u - y, y, m.z > 0.0);
        sy = select(w - 1u - x, x, m.y > 0.0);
    }
    return input_a[sy * source_desc().x + sx];
}

// A channel's 16-bit code for the integer mip average: (v.clamp(0, 1) * 65535).round() as
// u32, NaN to 0 as Rust's cast.
fn mip_code(v: f32) -> u32 {
    if is_nan(v) { return 0u; }
    return u32(round_away(barrier(clamp(v, 0.0, 1.0) * 65535.0)));
}

// One halving along the chosen axes: the average of each texel pair or quad, summed in the
// CPU's order ((p00 + p10) + p01) + p11 and divided exactly; 16-bit targets average the
// integer codes with ties up (the GPU mip generator rule the CPU measured). store() then
// quantizes to the level's format.
fn halve_pixel(k: u32) -> vec4<f32> {
    let w = config.header.z;
    let x = k % w;
    let y = k / w;
    let nx = 1u + ((config.header.w >> 8u) & 1u);
    let ny = 1u + ((config.header.w >> 9u) & 1u);
    let sw = source_desc().x;
    // The CPU starts from 0.0 and adds each texel: 0.0 + (-0.0) is +0.0. A literal zero may let
    // a compiler drop that first addition (and keep a -0.0; NVIDIA 615 does not, measured with
    // gpu_transform.py negzero_*), so start from a runtime zero.
    var acc = vec4<f32>(bitcast<f32>(config.header.y >> 31u));
    var sum = vec4<u32>(0u);
    for (var dy = 0u; dy < ny; dy++) {
        for (var dx = 0u; dx < nx; dx++) {
            let p = input_a[(y * ny + dy) * sw + x * nx + dx];
            acc = acc + p;
            sum = sum + vec4<u32>(mip_code(p.x), mip_code(p.y), mip_code(p.z), mip_code(p.w));
        }
    }
    let n = nx * ny;
    if config.options.x == 1u {
        let q = (sum + vec4<u32>(n / 2u)) / vec4<u32>(n);
        return vec4<f32>(unorm_float(q.x, 16u), unorm_float(q.y, 16u), unorm_float(q.z, 16u), unorm_float(q.w, 16u));
    }
    // n is 1, 2 or 4: exact reciprocals.
    var inv = 1.0;
    if n == 2u { inv = 0.5; }
    if n == 4u { inv = 0.25; }
    return acc * inv;
}

// One 2x upsample along the chosen axes: the texture unit's bilinear at the new pixel
// centres, edges clamped (tiled axes wrap), in the input descriptor's format.
fn upsample_pixel(k: u32) -> vec4<f32> {
    let w = config.header.z;
    let h = config.header.y / w;
    let u = (f32(k % w) + 0.5) / f32(w);
    let v = (f32(k / w) + 0.5) / f32(h);
    let wrap_x = (config.header.w & 1u) != 0u;
    let wrap_y = (config.header.w & 2u) != 0u;
    return sample(u, v, true, wrap_x, wrap_y, true, vec4<f32>(0.0));
}

fn transform_pixel(k: u32) -> vec4<f32> {
    let w = f32(config.header.z);
    let h = f32(config.header.y / config.header.z);
    let x = f32(k % config.header.z);
    let y = f32(k / config.header.z);
    // Output sizes are powers of two: the divisions are exact.
    let dx = (x + 0.5) / w - 0.5;
    let dy = (y + 0.5) / h - 0.5;
    let m = config.parameters[0];
    let off = config.parameters[1].xy;
    // ((m0*dx + m2*dy) + 0.5) + off.x, each operation rounded as on the CPU.
    let u = ((barrier(m.x * dx) + barrier(m.z * dy)) + 0.5) + off.x;
    let v = ((barrier(m.y * dx) + barrier(m.w * dy)) + 0.5) + off.y;
    let wrap_x = (config.header.w & 1u) != 0u;
    let wrap_y = (config.header.w & 2u) != 0u;
    let bilinear = (config.header.w & 4u) != 0u;
    let clamp_border = (config.header.w & 8u) != 0u;
    return sample(u, v, bilinear, wrap_x, wrap_y, clamp_border, config.extra);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.y * config.header.z + id.x;
    if id.x >= config.header.z || k >= config.header.y { return; }
    switch config.header.x {
        case 1u: { store(k, permute_pixel(k)); }
        case 2u: { store(k, halve_pixel(k)); }
        case 3u: { store(k, upsample_pixel(k)); }
        default: { store(k, transform_pixel(k)); }
    }
}
