// Shared helpers of the colour kernels (gradient map, curve, HSL, emboss, dynamic gradient):
// ops.rs concatenates this file after the prelude and before each kernel's own file. Every
// helper reproduces one Rust f32 operation of the CPU reference bit for bit (finite operands).
//
// Besides contracting a*b + c into an fma, the NVIDIA 615 compiler distributes a constant
// factor over a sum with a constant: 6.0 * (d + E) became fma(6, d, 6E), one ULP off
// (cx/gpu_color/probe/distribute.wgsl). The kernels therefore wrap in barrier() every sum
// that feeds a product, as well as every product that feeds a sum.
//
// The Rust helpers and the correctly rounded division (div_rn) live in the prelude, which every
// kernel is built on.

// --- 64-bit unsigned integers as vec2<u32>(low, high), for mix_f64 ---

fn u64_bitlen(v: vec2<u32>) -> u32 {
    if v.y != 0u { return 33u + firstLeadingBit(v.y); }
    if v.x != 0u { return 1u + firstLeadingBit(v.x); }
    return 0u;
}

// x << s for a 32-bit x, s in [0, 31].
fn u64_shl32(x: u32, s: u32) -> vec2<u32> {
    if s == 0u { return vec2<u32>(x, 0u); }
    return vec2<u32>(x << s, x >> (32u - s));
}

// v >> s, s in [0, 63].
fn u64_shr(v: vec2<u32>, s: u32) -> vec2<u32> {
    if s == 0u { return v; }
    if s >= 32u { return vec2<u32>(v.y >> (s - 32u), 0u); }
    return vec2<u32>((v.x >> s) | (v.y << (32u - s)), v.y >> s);
}

fn u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

// a - b for a >= b.
fn u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

fn u64_ge(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.y > b.y || (a.y == b.y && a.x >= b.x);
}

fn u64_bit(v: vec2<u32>, i: u32) -> u32 {
    if i >= 32u { return (v.y >> (i - 32u)) & 1u; }
    return (v.x >> i) & 1u;
}

// Whether any bit below position i (in [0, 64]) is set.
fn u64_low_set(v: vec2<u32>, i: u32) -> bool {
    if i == 0u { return false; }
    if i < 32u { return (v.x & ((1u << i) - 1u)) != 0u; }
    if i == 32u { return v.x != 0u; }
    return v.x != 0u || (v.y & ((1u << (i - 32u)) - 1u)) != 0u;
}

// v with the bits below position i (in [1, 63]) cleared, plus 2^i when `up`.
fn u64_cut(v: vec2<u32>, i: u32, up: bool) -> vec2<u32> {
    var r: vec2<u32>;
    var one: vec2<u32>;
    if i < 32u {
        r = vec2<u32>(v.x & ~((1u << i) - 1u), v.y);
        one = vec2<u32>(1u << i, 0u);
    } else {
        r = vec2<u32>(0u, v.y & ~((1u << (i - 32u)) - 1u));
        one = vec2<u32>(0u, 1u << (i - 32u));
    }
    if up { r = u64_add(r, one); }
    return r;
}

// v rounded to `keep` significant bits, ties to even (the scale is kept).
fn u64_round(v: vec2<u32>, keep: u32) -> vec2<u32> {
    let n = u64_bitlen(v);
    if n <= keep { return v; }
    let drop = n - keep;
    let guard = u64_bit(v, drop - 1u) != 0u;
    let up = guard && (u64_low_set(v, drop - 1u) || u64_bit(v, drop) != 0u);
    return u64_cut(v, drop, up);
}

// f32(f64(p0) * (1 - k/256) + f64(p1) * (k/256)) as ops.rs "sample_f32_lut" computes it, for
// k in [1, 255] and finite p0, p1 that are zero or at least 2^-95 in magnitude and whose
// exponents differ by at most 31 (ops.rs checks every texel pair before taking the GPU path).
// Each product is an exact integer of at most 32 bits times a power of two; the sum is formed
// exactly in 64-bit integer arithmetic, rounded to 53 bits (the f64 addition) and then to 24
// (the conversion to f32), both ties to even.
// Non-finite texels (the v8 table divides by zero between keys at one position) follow IEEE
// as the CPU computes them on x86-64: a NaN propagates its payload (quieted), infinities stay,
// opposite infinities give the default NaN 0xffc00000; ops.rs falls back when NaNs of two
// payloads are neighbours (which one propagates is up to the compiler's operand order).
fn mix_f64(p0: f32, p1: f32, k: u32) -> f32 {
    let b0 = bitcast<u32>(p0);
    let b1 = bitcast<u32>(p1);
    if !finite_bits(p0) || !finite_bits(p1) {
        if nan_bits(p0) { return bitcast<f32>(b0 | 0x00400000u); }
        if nan_bits(p1) { return bitcast<f32>(b1 | 0x00400000u); }
        if !finite_bits(p0) && !finite_bits(p1) && b0 != b1 { return bitcast<f32>(0xffc00000u); }
        if !finite_bits(p0) { return p0; }
        return p1;
    }
    var e0 = (b0 >> 23u) & 255u;
    var e1 = (b1 >> 23u) & 255u;
    if e0 == 0u && e1 == 0u {
        // (+-0)(1 - w) + (+-0) w: negative only when both zeros are
        return bitcast<f32>(b0 & b1 & 0x80000000u);
    }
    var x0 = ((b0 & 0x007fffffu) | 0x00800000u) * (256u - k);
    var x1 = ((b1 & 0x007fffffu) | 0x00800000u) * k;
    if e0 == 0u { x0 = 0u; e0 = e1; }
    if e1 == 0u { x1 = 0u; e1 = e0; }
    var a: vec2<u32>;
    var b: vec2<u32>;
    if e0 >= e1 {
        a = u64_shl32(x0, e0 - e1);
        b = vec2<u32>(x1, 0u);
    } else {
        a = vec2<u32>(x0, 0u);
        b = u64_shl32(x1, e1 - e0);
    }
    let low = min(e0, e1);
    let s0 = b0 & 0x80000000u;
    let s1 = b1 & 0x80000000u;
    var mag: vec2<u32>;
    var sign: u32;
    if s0 == s1 {
        mag = u64_add(a, b);
        sign = s0;
    } else if u64_ge(a, b) {
        mag = u64_sub(a, b);
        sign = s0;
    } else {
        mag = u64_sub(b, a);
        sign = s1;
    }
    if mag.x == 0u && mag.y == 0u { return 0.0; }
    let r = u64_round(u64_round(mag, 53u), 24u);
    // value = r * 2^(low - 158) = m * 2^(exponent - 150), m the top 24 bits of r
    let n = u64_bitlen(r);
    var m: u32;
    if n > 24u { m = u64_shr(r, n - 24u).x; } else { m = r.x << (24u - n); }
    let exponent = n + low - 32u;
    return bitcast<f32>(sign | (exponent << 23u) | (m & 0x007fffffu));
}

// --- the texture unit's bilinear of input_b, image.rs "bilinear_or" with border None ---
// config.selectors is input_b's descriptor (width, height, storage format, colour).

fn b_wrap(i: i32, n: i32) -> i32 {
    return ((i % n) + n) % n;
}

// image.rs "fetch": tiled axes wrap, the others clamp to the edge.
fn fetch_b(x: i32, y: i32, wrap_x: bool, wrap_y: bool) -> vec4<f32> {
    let w = i32(config.selectors.x);
    let h = i32(config.selectors.y);
    var xx = clamp(x, 0, w - 1);
    var yy = clamp(y, 0, h - 1);
    if wrap_x { xx = b_wrap(x, w); }
    if wrap_y { yy = b_wrap(y, h); }
    return input_b[u32(yy * w + xx)];
}

// The CPU's ((a*w0 + b*w1) + c*w2) + d*w3, each product and sum rounded.
fn b_weighted(a: f32, b: f32, c: f32, d: f32, wt: vec4<f32>) -> f32 {
    var o = barrier(a * wt.x);
    o = o + barrier(b * wt.y);
    o = o + barrier(c * wt.z);
    o = o + barrier(d * wt.w);
    return o;
}

// A UNORM texel's 16-bit codes, (v * 65535).round() as u32.
fn b_code16(v: vec4<f32>) -> vec4<u32> {
    return vec4<u32>(
        u32(rs_round(barrier(v.x * 65535.0))), u32(rs_round(barrier(v.y * 65535.0))),
        u32(rs_round(barrier(v.z * 65535.0))), u32(rs_round(barrier(v.w * 65535.0))));
}

// Coordinates within +-2^24 texels (ops.rs checks the parameters that reach them).
fn sample_b(u: f32, v: f32, wrap_x: bool, wrap_y: bool) -> vec4<f32> {
    let x = barrier(u * f32(config.selectors.x)) - 0.5;
    let y = barrier(v * f32(config.selectors.y)) - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = rs_round(barrier(x - x0) * 256.0) * (1.0 / 256.0);
    let fy = rs_round(barrier(y - y0) * 256.0) * (1.0 / 256.0);
    let cross = rs_round(fx * fy * 256.0) * (1.0 / 256.0);
    // barrier(): the weights are sums feeding products (exact here, but not to be distributed)
    let wt = vec4<f32>(barrier(1.0 - fx - fy + cross), barrier(fx - cross), barrier(fy - cross), cross);
    let xi = i32(x0);
    let yi = i32(y0);
    let a = fetch_b(xi, yi, wrap_x, wrap_y);
    let b = fetch_b(xi + 1, yi, wrap_x, wrap_y);
    let c = fetch_b(xi, yi + 1, wrap_x, wrap_y);
    let d = fetch_b(xi + 1, yi + 1, wrap_x, wrap_y);
    if config.selectors.z <= 1u {
        // UNORM sources interpolate 16-bit codes, ties up; the weights are multiples of 1/256
        let wq = vec4<u32>(wt * 256.0);
        let sum = b_code16(a) * wq.x + b_code16(b) * wq.y + b_code16(c) * wq.z + b_code16(d) * wq.w;
        let code = (sum + vec4<u32>(128u)) >> vec4<u32>(8u);
        return vec4<f32>(unorm_float(code.x, 16u), unorm_float(code.y, 16u), unorm_float(code.z, 16u), unorm_float(code.w, 16u));
    }
    return vec4<f32>(
        b_weighted(a.x, b.x, c.x, d.x, wt), b_weighted(a.y, b.y, c.y, d.y, wt),
        b_weighted(a.z, b.z, c.z, d.z, wt), b_weighted(a.w, b.w, c.w, d.w, wt));
}
