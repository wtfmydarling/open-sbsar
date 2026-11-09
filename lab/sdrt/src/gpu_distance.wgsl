// Distance (op 0x2A), ops.rs "distance" (the 14.1 mode: exact query positions, UNORM16 seeds,
// fused metric, SDRT_DISTQ / SDRT_DIST14 at their defaults). The CPU runs jump flooding: a pass
// per step (max(w, h) / 2 down to 1), each pixel reading the previous pass's "nearest mask
// pixel" of itself and of its 8 neighbours at +-step, in the order (1,0) (-1,0) (0,1) (0,-1)
// (1,1) (-1,-1) (1,-1) (-1,1), keeping the first strictly closer candidate. One GPU pass per
// step mirrors that order; the pass output holds the nearest pixel index (-1: none) in .x.
// The sides are powers of two (ops.rs falls back otherwise): every division below is exact.
// gpu_sample.wgsl is concatenated before this file (resize pass, s_ helpers).
//
// config.header.x selects the pass:
//   0 flood  - input_a the previous pass (or the mask: header.w bit 2); selectors.x = step,
//              selectors.y = max(w, h)
//   1 shade  - input_a as above, input_b the source image (header.w bit 6) at the node's size
//   2 resize - image.rs "resized" of input_a (the mask / source input_at), descriptor selectors
// config.header.w bits: 0 wraps x, 1 wraps y, 2 input_a is the mask (nearest = the pixel itself
// where the mask is > 0.5), 3 the mask is colour (alpha is read, else red), 4-5 metric (0
// Euclidean, 1 Manhattan, 2 Chebyshev), 6 a source image, 7 combine (flags & 2), 9 colour node.
// config.values: 1/ps.x, 1/ps.y, dmax = distance / 256, 1/dmax (computed on the CPU as the CPU
// function does).

fn wrap_x() -> bool { return (config.header.w & 1u) != 0u; }
fn wrap_y() -> bool { return (config.header.w & 2u) != 0u; }

// The nearest mask pixel index stored at pixel idx of input_a, or -1.
fn near_at(idx: u32) -> i32 {
    let p = input_a[idx];
    if (config.header.w & 4u) != 0u {
        var m = p.x;
        if (config.header.w & 8u) != 0u { m = p.w; }
        if m > 0.5 { return i32(idx); }
        return -1;
    }
    return i32(p.x);
}

// ((v + 0.5) / n * 65535).round() in f64, as the CPU's `code`: (2v + 1) * 65535 / (2n) rounded
// half up, in integers (n a power of two <= 16384: the product stays below 2^32).
fn code16(v: u32, n: u32) -> u32 {
    return ((2u * v + 1u) * 65535u + n) / (2u * n);
}

// ops.rs delta14: the pixel centre minus the seed's UNORM16 position, tiled axes folded to
// +-0.5.
fn delta14(p: u32, s: u32, n: u32, wrap: bool) -> f32 {
    var d = (f32(p) + 0.5) / f32(n) - unorm_float(code16(s, n), 16u);
    if wrap && abs(d) > 0.5 { d -= select(-1.0, 1.0, d > 0.0); }
    return d;
}

// qd2 of the CPU for pixel k and seed s: the squared distance (Euclidean: du.mul_add(du,
// dv * dv)), the Manhattan sum or the Chebyshev maximum. The Euclidean root is taken in
// `shade` only.
fn metric(k: u32, s: u32) -> f32 {
    let w = config.header.z;
    let h = config.header.y / w;
    let du = barrier(delta14(k % w, s % w, w, wrap_x()) * config.values.x);
    let dv = barrier(delta14(k / w, s / w, h, wrap_y()) * config.values.y);
    let mode = (config.header.w >> 4u) & 3u;
    if mode == 0u { return fma(du, du, barrier(dv * dv)); }
    if mode == 1u { return abs(du) + abs(dv); }
    return max(abs(du), abs(dv));
}

// A jump position along one axis (the CPU's `at`): from the pixel centre, d * step / max(w, h)
// in normalized coordinates, floor of uv * n, rem_euclid on a tiled axis (n a power of two) and
// clamped otherwise.
fn jump_axis(p: u32, d: i32, n: u32, wrap: bool) -> u32 {
    let uv = (f32(p) + 0.5) / f32(n) + f32(d) * f32(config.selectors.x) / f32(config.selectors.y);
    let v = i32(floor(uv * f32(n)));
    if wrap { return u32(v & i32(n - 1u)); }
    return u32(clamp(v, 0, i32(n) - 1));
}

fn flood_pixel(k: u32) -> i32 {
    let w = config.header.z;
    let h = config.header.y / w;
    let x = k % w;
    let y = k / w;
    let cur = near_at(k);
    var out = cur;
    var have = cur >= 0;
    var best = 0.0;
    if have { best = metric(k, u32(cur)); }
    for (var i = 0u; i < 8u; i++) {
        var dx = 0;
        var dy = 0;
        switch i {
            case 0u: { dx = 1; }
            case 1u: { dx = -1; }
            case 2u: { dy = 1; }
            case 3u: { dy = -1; }
            case 4u: { dx = 1; dy = 1; }
            case 5u: { dx = -1; dy = -1; }
            case 6u: { dx = 1; dy = -1; }
            default: { dx = -1; dy = 1; }
        }
        let ax = jump_axis(x, dx, w, wrap_x());
        let ay = jump_axis(y, dy, h, wrap_y());
        let q = near_at(ay * w + ax);
        if q < 0 { continue; }
        let d = metric(k, u32(q));
        if !have || d < best {
            best = d;
            out = q;
            have = true;
        }
    }
    return out;
}

// The correctly rounded square root (Rust's f32::sqrt) of x >= 0: the GPU's sqrt is within a
// few ulps; each step moves to the neighbour that is nearer in the root, decided exactly: the
// root's midpoint m = (c + n) / 2 has m^2 = (c^2 + n^2) / 2 - ulp^2 / 4, so x is above it
// exactly when x - c^2 >= n^2 - x (both residuals are exact fma results, integer multiples
// of ulp^2; no tie is possible between a float and a midpoint).
fn exact_sqrt(x: f32) -> f32 {
    if !(x > 0.0) { return 0.0; }
    var s = sqrt(x);
    for (var i = 0; i < 3; i++) {
        let up = bitcast<f32>(bitcast<u32>(s) + 1u);
        if fma(-s, s, x) >= fma(up, up, -x) { s = up; } else { break; }
    }
    for (var i = 0; i < 3; i++) {
        let dn = bitcast<f32>(bitcast<u32>(s) - 1u);
        if fma(-dn, dn, x) < fma(s, s, -x) { s = dn; } else { break; }
    }
    return s;
}

fn shade_pixel(k: u32) -> vec4<f32> {
    let nk = near_at(k);
    var within = false;
    var v = 0.0;
    if nk >= 0 {
        // a mask pixel is at distance 0 from itself (not the quantized seed offset)
        var d = 0.0;
        if nk != i32(k) {
            d = metric(k, u32(nk));
            if ((config.header.w >> 4u) & 3u) == 0u { d = exact_sqrt(d); }
        }
        within = d < config.values.z;
        if config.values.z != 0.0 {
            v = max(fma(-d, config.values.w, 1.0), 0.0);
        }
    }
    if (config.header.w & 64u) == 0u { return vec4<f32>(v, v, v, 1.0); }
    let here = input_b[k];
    if (config.header.w & 128u) != 0u {
        // combine: the falloff goes to alpha only for colour
        var p = here;
        if nk >= 0 { p = input_b[nk]; }
        var q = here;
        if within { q = p; }
        if (config.header.w & 512u) != 0u { return vec4<f32>(q.x, q.y, q.z, p.w * v); }
        return p * v;
    }
    if within { return input_b[nk]; }
    return here;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.y * config.header.z + id.x;
    if id.x >= config.header.z || k >= config.header.y { return; }
    switch config.header.x {
        case 1u: { store(k, shade_pixel(k)); }
        case 2u: { store(k, s_resize_pixel(k)); }
        default: { store(k, vec4<f32>(f32(flood_pixel(k)), 0.0, 0.0, 0.0)); }
    }
}
