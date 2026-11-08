// Blur (op 0x14), ops.rs "blur": two separable passes of the reference GPU's box taps
// ("blur_taps", per output parity, "box_pass_parity"); a larger input's first pass instead
// fetches it at its own size with "bilinear_or" at the paired taps' fractional offsets
// ("blur_fetches"). The helpers (s_*) are in gpu_sample.wgsl; ops.rs try_gpu_blur computes the
// taps on the CPU and binds them as input_b, one vec4 (offset, weight, 0, 0) each.
//
// config.header.x selects the pass:
//   0 box      - box_pass_parity along x (or y: header.w bit 8) over input_a; offsets are texels
//   1 fetches  - the larger input's horizontal pass: offsets are u steps (pixels / node width)
//   2 resize   - image.rs "resized" of input_a (input_at)
// bitcast(config.values) = (even start, even count, odd start, odd count) in input_b: the taps
// of an output pixel whose coordinate along the pass is even / odd. config.selectors is
// input_a's descriptor. config.header.w bits 0 / 1 wrap x / y.

fn wrap_bits() -> vec2<bool> {
    return vec2<bool>((config.header.w & 1u) != 0u, (config.header.w & 2u) != 0u);
}

// This pixel's taps: (start, count) by the parity of `along`.
fn taps_of(along: u32) -> vec2<u32> {
    let t = bitcast<vec4<u32>>(config.values);
    if along % 2u == 0u { return t.xy; }
    return t.zw;
}

fn box_pixel(k: u32) -> vec4<f32> {
    let w = config.header.z;
    let x = k % w;
    let y = k / w;
    let along_y = (config.header.w & 0x100u) != 0u;
    let taps = taps_of(select(x, y, along_y));
    let wrap = wrap_bits();
    // acc[c] += p[c] * wt from 0.0, in tap order
    var acc = s_zero4();
    for (var i = 0u; i < taps.y; i++) {
        let e = input_b[taps.x + i];
        let t = i32(e.x);
        var p: vec4<f32>;
        if along_y { p = s_fetch(0u, config.selectors, i32(x), i32(y) + t, wrap); }
        else { p = s_fetch(0u, config.selectors, i32(x) + t, i32(y), wrap); }
        acc = acc + barrier4(p * e.y);
    }
    return acc;
}

fn fetches_pixel(k: u32) -> vec4<f32> {
    let w = config.header.z;
    let h = config.header.y / w;
    let x = k % w;
    let y = k / w;
    let u = (f32(x) + 0.5) / f32(w);
    let v = (f32(y) + 0.5) / f32(h);
    let taps = taps_of(x);
    let wrap = wrap_bits();
    var acc = s_zero4();
    for (var i = 0u; i < taps.y; i++) {
        let e = input_b[taps.x + i];
        acc = acc + barrier4(s_bilinear_or(0u, config.selectors, u + e.x, v, wrap) * e.y);
    }
    return acc;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.y * config.header.z + id.x;
    if id.x >= config.header.z || k >= config.header.y { return; }
    switch config.header.x {
        case 1u: { store(k, fetches_pixel(k)); }
        case 2u: { store(k, s_resize_pixel(k)); }
        default: { store(k, box_pixel(k)); }
    }
}
