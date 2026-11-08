// Gradient map (op 0x00): ops.rs "gradient_map" maps each input value (a colour input's
// alpha, else its gray value; non-finite values read as 0) through a look-up table that
// ops.rs try_gpu_gradient_map builds on the CPU exactly as the CPU path does and binds as
// input_b. config.header.x selects the table:
//   0  1024 RGBA f32 texels ("sample_f32_lut"): coordinate fma(x, 1023/1024, 0.5/1024) (Rust's
//      mul_add; the driver's fma is fused), the 1/256 weight floor(f*256 + 0.5) in f32, the
//      mix in f64 (mix_f64); half-float nodes truncate toward zero (image.rs "half_trunc").
//   1  256 RGBA UNORM16 codes stored as exact floats: the same coordinate rule with 256
//      texels, the weight rounded half up exactly (the CPU adds 0.5 in f64), the integer
//      interpolation ties up, then code/65535 correctly rounded.
// config.header.w bits: 0 repeat (the coordinate wraps, and so does the texel pair),
// 1 half-float truncation, 2 the input is colour (its alpha is mapped).

fn gm_half_trunc(v: f32) -> f32 {
    if !finite_bits(v) { return v; }
    let bits = bitcast<u32>(v);
    let sign = bits & 0x80000000u;
    let e = i32((bits >> 23u) & 255u) - 127;
    if e > 15 { return bitcast<f32>(sign | 0x477fe000u); }
    if e < -24 { return bitcast<f32>(sign); }
    var keep = 10;
    if e < -14 { keep = 10 - (-14 - e); }
    return bitcast<f32>(bits & ~((1u << u32(23 - keep)) - 1u));
}

fn gm_f32(x: f32, repeat: bool) -> vec4<f32> {
    var coord: f32;
    if repeat {
        coord = rs_rem1(fma(x, 1023.0 / 1024.0, 0.5 / 1024.0));
    } else {
        coord = fma(rs_clamp(x, 0.0, 1.0), 1023.0 / 1024.0, 0.5 / 1024.0);
    }
    let u = barrier(barrier(coord) * 1024.0) - 0.5;
    let fi = floor(u);
    let i = i32(fi);
    let k = u32(floor(barrier(barrier(u - fi) * 256.0) + 0.5));
    var i0 = clamp(i, 0, 1023);
    var i1 = clamp(i + 1, 0, 1023);
    if repeat {
        i0 = ((i % 1024) + 1024) % 1024;
        i1 = ((i + 1) % 1024 + 1024) % 1024;
    }
    let p0 = input_b[u32(i0)];
    let p1 = input_b[u32(i1)];
    if k == 0u { return p0; }
    if k >= 256u { return p1; }
    return vec4<f32>(mix_f64(p0.x, p1.x, k), mix_f64(p0.y, p1.y, k), mix_f64(p0.z, p1.z, k), mix_f64(p0.w, p1.w, k));
}

fn gm_unorm16(x: f32, repeat: bool) -> vec4<f32> {
    var i0: u32;
    var i1: u32;
    var f: f32;
    if repeat {
        let coord = rs_rem1(fma(x, 255.0 / 256.0, 0.5 / 256.0));
        let u = barrier(barrier(coord) * 256.0) - 0.5;
        if u < 0.0 {
            i0 = 255u; i1 = 0u; f = u + 1.0;
        } else if u >= 255.0 {
            i0 = 255u; i1 = 0u; f = u - 255.0;
        } else {
            let fi = floor(u);
            i0 = u32(fi); i1 = i0 + 1u; f = u - fi;
        }
    } else {
        let coord = fma(rs_clamp(x, 0.0, 1.0), 255.0 / 256.0, 0.5 / 256.0);
        let u = barrier(barrier(coord) * 256.0) - 0.5;
        let fi = max(floor(u), 0.0);
        i0 = min(u32(fi), 254u); i1 = i0 + 1u; f = u - f32(i0);
    }
    let wt = u32(clamp(floor_half_up(barrier(f) * 256.0), 0.0, 256.0));
    let t0 = vec4<u32>(input_b[i0]);
    let t1 = vec4<u32>(input_b[i1]);
    let code = (t0 * (256u - wt) + t1 * wt + vec4<u32>(128u)) >> vec4<u32>(8u);
    return vec4<f32>(unorm_float(code.x, 16u), unorm_float(code.y, 16u), unorm_float(code.z, 16u), unorm_float(code.w, 16u));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.y * config.header.z + id.x;
    if id.x >= config.header.z || k >= config.header.y { return; }
    let s = input_a[k];
    var x = s.x;
    if (config.header.w & 4u) != 0u { x = s.w; }
    if !finite_bits(x) { x = 0.0; }
    let repeat = (config.header.w & 1u) != 0u;
    var c: vec4<f32>;
    if config.header.x == 1u {
        c = gm_unorm16(x, repeat);
    } else {
        c = gm_f32(x, repeat);
        if (config.header.w & 2u) != 0u {
            c = vec4<f32>(gm_half_trunc(c.x), gm_half_trunc(c.y), gm_half_trunc(c.z), gm_half_trunc(c.w));
        }
    }
    store(k, c);
}
