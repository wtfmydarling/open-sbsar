// Curve (op 0x2C): ops.rs "curve" evaluates each Bézier curve through a table of 1025 values
// at x = i/1024, built on the CPU by "bezier_curve" (ops.rs try_gpu_curve uploads that very
// table as input_b: curve c, entry i at scalar index c*1025 + i). Per channel:
//   t = clamp(x, 0, 1) * 1024, i = min(floor(t), 1023), f = round((t - i) * 256) / 256,
//   l[i] + (l[i+1] - l[i]) * f
// and an empty curve is the identity. A gray node applies curve 0; a colour node applies the
// luminance curve 0 to R, G, B, then curves 1..4 to R, G, B, alpha.
// config.header.w bits: 0 the input is not read, the curve runs along x ((x + 0.5)/w),
// 1 colour node, 8 + c curve c is empty.

fn curve_entry(index: u32) -> f32 {
    return input_b[index >> 2u][index & 3u];
}

fn curve_eval(c: u32, x: f32) -> f32 {
    if ((config.header.w >> (8u + c)) & 1u) != 0u { return x; }
    let t = barrier(rs_clamp(x, 0.0, 1.0)) * 1024.0;
    let i = min(u32(floor(t)), 1023u);
    let f = rs_round(barrier(t - f32(i)) * 256.0) * (1.0 / 256.0);
    let a = curve_entry(c * 1025u + i);
    let b = curve_entry(c * 1025u + i + 1u);
    return a + barrier(barrier(b - a) * f);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = config.header.z;
    let k = id.y * w + id.x;
    if id.x >= w || k >= config.header.y { return; }
    var s: vec4<f32>;
    if (config.header.w & 1u) != 0u {
        // output sizes are powers of two: the division is exact
        let t = barrier(f32(id.x) + 0.5) / f32(w);
        s = vec4<f32>(t);
    } else {
        s = input_a[k];
    }
    if (config.header.w & 2u) == 0u {
        store(k, vec4<f32>(curve_eval(0u, s.x)));
        return;
    }
    var c = s;
    c.x = curve_eval(0u, c.x);
    c.y = curve_eval(0u, c.y);
    c.z = curve_eval(0u, c.z);
    c.x = curve_eval(1u, c.x);
    c.y = curve_eval(2u, c.y);
    c.z = curve_eval(3u, c.z);
    c.w = curve_eval(4u, c.w);
    store(k, c);
}
