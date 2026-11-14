// Sampling helpers shared by the filter kernels (gpu_warp.wgsl, gpu_blur.wgsl, gpu_mblur.wgsl;
// ops.rs concatenates prelude + this file + the kernel). They reproduce image.rs bit for bit:
//   s_fetch         - "fetch": tiled axes wrap (rem_euclid), the others clamp;
//   s_bilinear_256  - "bilinear_256": 1/256 fractions, lerps (the default "bilinear");
//   s_bilinear_or   - "bilinear_or" with border None: 1/256 fractions, the coupled cross weight,
//                     UNORM textures (format <= 1) interpolated as 16-bit codes;
//   s_resize_pixel  - "resized" (default mode: clamped edges, "bilinear").
// An image is an input slot (0: input_a, 1: input_b) and a descriptor (width, height, sampled
// storage format, colour flag). Coordinates may be anything: the warps compute them from texel
// values. Huge, infinite and NaN coordinates index texels as the CPU's `x0 as i64` (saturating,
// NaN to 0) and `x0 + 1` (wrapping) do; NaN weights are 0 in the integer path as Rust's casts.
// Every product that feeds an addition goes through barrier()/barrier4(), so each multiply and
// add rounds on its own as on the CPU.


// The CPU's 0.0 accumulator start (0.0 + -0.0 is +0.0): a zero the compiler cannot fold away.
fn s_zero4() -> vec4<f32> {
    return vec4<f32>(bitcast<f32>(config.header.y >> 31u));
}

fn s_texel(slot: u32, i: u32) -> vec4<f32> {
    if slot == 0u { return input_a[i]; }
    return input_b[i];
}

fn s_is_nan(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;
}

// Rust's f32::round (half away from zero) for every input: trunc, then the exact remainder.
// (floor(v + 0.5) rounds 0.49999997 + 0.5 up to 1.)
fn s_round(v: f32) -> f32 {
    let t = trunc(v);
    if abs(v - t) >= 0.5 {
        return t + select(-1.0, 1.0, v > 0.0);
    }
    return t;
}

// Rust's `v as u32`: truncation, NaN and negatives to 0, saturating at u32::MAX.
fn s_as_u32(v: f32) -> u32 {
    if s_is_nan(v) || v <= 0.0 { return 0u; }
    if v >= 4294967296.0 { return 0xffffffffu; }
    return u32(v);
}

// q / 65535 correctly rounded for any q < 2^24 (the prelude's unorm_float covers q <= 65535):
// q = a*65535 + b, so q/65535 is a followed by the binary fraction b repeating every 16 bits.
// A repeating non-zero fraction is never a tie: the bit after the last kept one rounds.
fn s_unorm16(q: u32) -> f32 {
    if q <= 65535u { return unorm_float(q, 16u); }
    let a = q / 65535u;
    let b = q % 65535u;
    if b == 0u { return f32(a); }
    let k = firstLeadingBit(a);
    let rep = (b << 16u) | b;
    let fb = 23u - k;
    var mant = (a << fb) | (rep >> (32u - fb));
    mant += (rep >> (31u - fb)) & 1u;
    return f32(mant) * bitcast<f32>((127u - fb) << 23u);
}

// 2^e mod n (n <= 65536).
fn s_pow2_mod(e: u32, n: u32) -> u32 {
    var p = 1u % n;
    for (var i = 0u; i < e; i++) { p = (p * 2u) % n; }
    return p;
}

// i.rem_euclid(n) for n > 0. Never `%` on a negative i32: on wgpu 0.19 / NVIDIA 615 it gives the
// unsigned remainder of the bit pattern (-5 % 3 == 2: 4294967291 % 3), right only for powers of
// two (probe_helpers.py).
fn s_wrap_i32(i: i32, n: i32) -> i32 {
    if i >= 0 { return i % n; }
    // -(i + 1) >= 0 for every i < 0 (no overflow at i32::MIN)
    return n - 1 - i32(u32(-(i + 1)) % u32(n));
}

fn s_axis_i32(i: i32, n: u32, wrap: bool) -> u32 {
    if wrap { return u32(s_wrap_i32(i, i32(n))); }
    return u32(clamp(i, 0, i32(n) - 1));
}

// The two texels of a bilinear lookup along one axis of n texels (n <= 65536): i = x0 as i64
// (saturating, NaN 0) and i + 1 (wrapping at i64::MAX), each rem_euclid(n) on a tiled axis and
// clamped to [0, n) otherwise. x0 is a floor()ed coordinate: integral, infinite or NaN.
fn s_axis_pair(x0: f32, n: u32, wrap: bool) -> vec2<u32> {
    let bits = bitcast<u32>(x0);
    let mag = bits & 0x7fffffffu;
    let neg = (bits >> 31u) != 0u;
    if mag > 0x7f800000u {
        return vec2<u32>(s_axis_i32(0, n, wrap), s_axis_i32(1, n, wrap));
    }
    if mag < 0x4e800000u {
        // |x0| < 2^30: exact in i32
        let i = i32(x0);
        return vec2<u32>(s_axis_i32(i, n, wrap), s_axis_i32(i + 1, n, wrap));
    }
    if !wrap {
        if neg { return vec2<u32>(0u, 0u); }
        // i64::MAX + 1 wraps to i64::MIN: that texel clamps to 0
        if mag >= 0x5f000000u { return vec2<u32>(n - 1u, 0u); }
        return vec2<u32>(n - 1u, n - 1u);
    }
    let m63 = s_pow2_mod(63u, n);
    if mag >= 0x5f000000u {
        // saturated (|x0| >= 2^63, infinities): i64::MIN, or i64::MAX whose successor wraps
        if neg {
            let r = (n - m63) % n;
            return vec2<u32>(r, (r + 1u) % n);
        }
        return vec2<u32>((m63 + n - 1u) % n, (n - m63) % n);
    }
    // 2^30 <= |x0| < 2^63: |x0| = mantissa * 2^(e - 23), reduced modulo n in u32
    let e = (mag >> 23u) - 127u;
    let mant = (mag & 0x7fffffu) | 0x800000u;
    let a = ((mant % n) * s_pow2_mod(e - 23u, n)) % n;
    var r = a;
    if neg { r = (n - a) % n; }
    return vec2<u32>(r, (r + 1u) % n);
}

// image.rs "fetch" at integer texel (x, y) (in i32 range).
fn s_fetch(slot: u32, d: vec4<u32>, x: i32, y: i32, wrap: vec2<bool>) -> vec4<f32> {
    return s_texel(slot, s_axis_i32(y, d.y, wrap.y) * d.x + s_axis_i32(x, d.x, wrap.x));
}

struct Quad {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    d: vec4<f32>,
    fx: f32,
    fy: f32,
}

// The four texels and the 1/256 fractions of a bilinear lookup at (u, v):
//   x = u*w - 0.5, x0 = floor(x), fx = ((x - x0)*256).round()/256, texels (x0|x0+1, y0|y0+1).
fn s_quad(slot: u32, d: vec4<u32>, u: f32, v: f32, wrap: vec2<bool>) -> Quad {
    let x = barrier(u * f32(d.x)) - 0.5;
    let y = barrier(v * f32(d.y)) - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = s_round((x - x0) * 256.0) * (1.0 / 256.0);
    let fy = s_round((y - y0) * 256.0) * (1.0 / 256.0);
    let px = s_axis_pair(x0, d.x, wrap.x);
    let py = s_axis_pair(y0, d.y, wrap.y);
    return Quad(s_texel(slot, py.x * d.x + px.x), s_texel(slot, py.x * d.x + px.y),
                s_texel(slot, py.y * d.x + px.x), s_texel(slot, py.y * d.x + px.y), fx, fy);
}

// image.rs "bilinear_256": top = a + (b - a)*fx, bot = c + (d - c)*fx, top + (bot - top)*fy.
fn s_bilinear_256(slot: u32, d: vec4<u32>, u: f32, v: f32, wrap: vec2<bool>) -> vec4<f32> {
    let q = s_quad(slot, d, u, v, wrap);
    let top = q.a + barrier4((q.b - q.a) * q.fx);
    let bot = q.c + barrier4((q.d - q.c) * q.fx);
    return top + barrier4((bot - top) * q.fy);
}

// A texel channel's 16-bit code: (v * 65535.0).round() as u32.
fn s_code16(v: f32) -> u32 {
    return s_as_u32(s_round(barrier(v * 65535.0)));
}

// image.rs "bilinear_or" with border None; d.z is the sampled texture format.
fn s_bilinear_or(slot: u32, d: vec4<u32>, u: f32, v: f32, wrap: vec2<bool>) -> vec4<f32> {
    let q = s_quad(slot, d, u, v, wrap);
    let cross = s_round(q.fx * q.fy * 256.0) * (1.0 / 256.0);
    let wt = vec4<f32>(1.0 - q.fx - q.fy + cross, q.fx - cross, q.fy - cross, cross);
    if d.z <= 1u {
        // UNORM: 16-bit codes times integer weights ((w * 256) as u32), wrapping u32 sums as
        // Rust's release build, ties up, then code / 65535 correctly rounded.
        let wq = vec4<u32>(s_as_u32(wt.x * 256.0), s_as_u32(wt.y * 256.0), s_as_u32(wt.z * 256.0), s_as_u32(wt.w * 256.0));
        var o: vec4<f32>;
        for (var c = 0u; c < 4u; c++) {
            let sum = s_code16(q.a[c]) * wq.x + s_code16(q.b[c]) * wq.y + s_code16(q.c[c]) * wq.z + s_code16(q.d[c]) * wq.w;
            o[c] = s_unorm16((sum + 128u) >> 8u);
        }
        return o;
    }
    // ((a*w0 + b*w1) + c*w2) + d*w3
    var o = barrier4(q.a * wt.x);
    o = o + barrier4(q.b * wt.y);
    o = o + barrier4(q.c * wt.z);
    o = o + barrier4(q.d * wt.w);
    return o;
}

// a / b correctly rounded for finite a and b > 0 with a normal (or zero) quotient: the GPU's
// division is not, so the quotient moves to the neighbour with the smaller exact fma remainder
// (a quotient of floats is never a tie).
fn s_div(a: f32, b: f32) -> f32 {
    var q = a / b;
    for (var i = 0; i < 8; i++) {
        let r = fma(-q, b, a);
        if r == 0.0 { break; }
        // the neighbour toward a/b: up when the remainder is positive
        let up = r > 0.0;
        var bits = bitcast<u32>(q);
        if (q > 0.0) == up { bits += 1u; } else { bits -= 1u; }
        let n = bitcast<f32>(bits);
        if abs(fma(-n, b, a)) < abs(r) { q = n; } else { break; }
    }
    return q;
}

// image.rs "resized" / "resized_fmt" (default mode): UNORM formats <= 1 interpolate in 16-bit
// integers with coupled 1/256 weights ("bilinear_or"), float formats use "bilinear" at the
// output's pixel centres, edges clamped. The output size is the pass size (powers of two).
fn s_resize_pixel(k: u32) -> vec4<f32> {
    let w = config.header.z;
    let h = config.header.y / w;
    let u = (f32(k % w) + 0.5) / f32(w);
    let v = (f32(k / w) + 0.5) / f32(h);
    if config.selectors.z <= 1u {
        return s_bilinear_or(0u, config.selectors, u, v, vec2<bool>(false, false));
    }
    return s_bilinear_256(0u, config.selectors, u, v, vec2<bool>(false, false));
}
