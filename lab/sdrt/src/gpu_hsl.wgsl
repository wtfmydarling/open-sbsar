// HSL (op 0x1C): ops.rs "hsl", "rgb_to_hsl" and "hsl_to_rgb" per pixel, every operation in the
// CPU's order and rounding: products feeding additions and sums feeding products go through
// barrier(), the two divisions by non-powers of two through the correctly rounded div_rn,
// % 1.0 through the exact rs_rem1. The output is colour (the CPU builds a colour image),
// alpha passes through.
// config.values: the hue, saturation and luminosity shifts as the CPU computes them per pixel,
// (hue - 0.5) * 2, 2 * (saturation - 0.5), 2 * (luminosity - 0.5); config.extra: Rust's f32
// constants 1/3, 1/6 and 2/3 (bit patterns from ops.rs, not re-folded by the WGSL compiler).
// Non-finite inputs are not reproduced (NaN payloads, the GPU's max/min/clamp on infinities).

const HSL_E: f32 = 1.0 / 4096.0;

fn hsl_channel(t0: f32, p: f32, q: f32) -> f32 {
    let t = barrier(rs_rem1(t0));
    // barrier() on the differences too: the driver distributes a product over a sum
    // (c * (a + b) -> c*a + c*b, see gpu_color_common.wgsl)
    let qp = barrier(q - p);
    if t < config.extra.y {
        return p + barrier(barrier(qp * 6.0) * t);
    }
    if t < 0.5 {
        return q;
    }
    if t < config.extra.z {
        return p + barrier(barrier(qp * barrier(config.extra.z - t)) * 6.0);
    }
    return p;
}

fn hsl_pixel(c: vec4<f32>) -> vec4<f32> {
    let r = c.x;
    let g = c.y;
    let b = c.z;
    // rgb_to_hsl
    let mx = rs_max(rs_max(r, g), b);
    let mn = rs_min(rs_min(r, g), b);
    var l = barrier(mx + mn) * 0.5;
    var h = 0.0;
    var s = 0.0;
    if mx != mn {
        let d = mx - mn;
        s = div_rn(d, (1.0 - abs(barrier(2.0 * l) - 1.0)) + HSL_E);
        var raw: f32;
        if mx == r {
            var add = 0.0;
            if g < b { add = barrier(6.0 * d); }
            raw = (g - b) + add;
        } else if mx == g {
            raw = (b - r) + barrier(2.0 * d);
        } else {
            raw = (r - g) + barrier(4.0 * d);
        }
        var bias = 2.0 * HSL_E;
        if raw == 0.0 {
            bias = -4.0 * HSL_E;
        } else if raw < 2.0 * d {
            bias = -2.0 * HSL_E;
        } else if raw <= 4.0 * d {
            bias = 0.0;
        }
        h = rs_rem1(div_rn(raw + bias, 6.0 * barrier(d + HSL_E)));
    }
    // the node's shifts
    h = rs_rem1(h + config.values.x);
    s = rs_clamp(s + config.values.y, 0.0, 1.0);
    l = rs_clamp(l + config.values.z, 0.0, 1.0);
    // hsl_to_rgb
    if s == 0.0 {
        return vec4<f32>(l, l, l, c.w);
    }
    var q: f32;
    if l < 0.5 {
        q = barrier(l * barrier(1.0 + s));
    } else {
        q = (l + s) - barrier(l * s);
    }
    let p = barrier(2.0 * l) - q;
    return vec4<f32>(hsl_channel(h + config.extra.x, p, q), hsl_channel(h, p, q), hsl_channel(h - config.extra.x, p, q), c.w);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.y * config.header.z + id.x;
    if id.x >= config.header.z || k >= config.header.y { return; }
    store(k, hsl_pixel(input_a[k]));
}
