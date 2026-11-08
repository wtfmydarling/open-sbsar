// Directional motion blur (op 0x16), ops.rs "motion_blur". The helpers (s_*) are in
// gpu_sample.wgsl; ops.rs try_gpu_motion_blur computes the taps, their offsets and the shear of
// every row (column) on the CPU, exactly as the CPU function does, and binds them as input_b.
//
// config.header.x selects the pass:
//   0 point    - one, two or three taps, or an axis-aligned blur: each tap a "bilinear" sample
//                at (x + 0.5 + ox, y + 0.5 + oy) / size; input_b taps (ox, oy, weight, 0)
//   1 shear    - the sheared image continued past the seam, blurred along the major axis: each
//                tap a "bilinear_or" sample of input_a at its own sheared position; input_b taps
//                (t, weight, 0, 0); the output is the (padded) blurred image
//   2 unshear  - "bilinear_or" of the blurred image (input_a) back at the unsheared position
//   3 resize   - image.rs "resized" of input_a (input_at)
//   4 copy     - input_a itself (no direction: the input is returned)
// bitcast(config.values) = (taps start, taps count, shear start, K): the shear of major-axis
// index j (-K <= j < size + K) is input_b[shear start + j + K].x. bitcast(config.extra).x is the
// padding ep of an untiled minor axis. config.header.w bits: 0 / 1 wrap x / y, 2 y is the major
// axis. config.selectors is input_a's descriptor.

fn wrap_bits() -> vec2<bool> {
    return vec2<bool>((config.header.w & 1u) != 0u, (config.header.w & 2u) != 0u);
}

fn point_pixel(k: u32) -> vec4<f32> {
    let w = config.header.z;
    let h = config.header.y / w;
    let x = f32(k % w) + 0.5;
    let y = f32(k / w) + 0.5;
    let t = bitcast<vec4<u32>>(config.values);
    let wrap = wrap_bits();
    var acc = s_zero4();
    for (var i = 0u; i < t.y; i++) {
        let e = input_b[t.x + i];
        let u = (x + e.x) / f32(w);
        let v = (y + e.y) / f32(h);
        acc = acc + barrier4(s_bilinear_256(0u, config.selectors, u, v, wrap) * e.z);
    }
    return acc;
}

fn shear_pixel(k: u32) -> vec4<f32> {
    let bw = config.header.z;
    let bx = k % bw;
    let by = k / bw;
    let t = bitcast<vec4<u32>>(config.values);
    let ep = f32(bitcast<vec4<u32>>(config.extra).x);
    let d = config.selectors;
    // the node's size: the input is at it
    let w = f32(d.x);
    let h = f32(d.y);
    let y_major = (config.header.w & 4u) != 0u;
    let wrap = wrap_bits();
    var acc = s_zero4();
    for (var i = 0u; i < t.y; i++) {
        let e = input_b[t.x + i];
        var u: f32;
        var v: f32;
        if y_major {
            let xx = f32(bx) - ep;
            let yy = f32(by) + e.x;
            let sh = input_b[t.z + u32(i32(by) + i32(e.x) + i32(t.w))].x;
            u = ((xx + 0.5) + sh) / w;
            v = (yy + 0.5) / h;
        } else {
            let xx = f32(bx) + e.x;
            let yy = f32(by) - ep;
            let sh = input_b[t.z + u32(i32(bx) + i32(e.x) + i32(t.w))].x;
            u = (xx + 0.5) / w;
            v = ((yy + 0.5) + sh) / h;
        }
        acc = acc + barrier4(s_bilinear_or(0u, d, u, v, wrap) * e.y);
    }
    return acc;
}

fn unshear_pixel(k: u32) -> vec4<f32> {
    let w = config.header.z;
    let h = config.header.y / w;
    let x = k % w;
    let y = k / w;
    let t = bitcast<vec4<u32>>(config.values);
    let epi = bitcast<vec4<u32>>(config.extra).x;
    let ep = f32(epi);
    let d = config.selectors;
    var u: f32;
    var v: f32;
    if (config.header.w & 4u) != 0u {
        let s = input_b[t.z + y + t.w].x;
        if epi == 0u { u = ((f32(x) + 0.5) - s) / f32(w); }
        else { u = s_div(((f32(x) + ep) + 0.5) - s, f32(d.x)); }
        v = (f32(y) + 0.5) / f32(h);
    } else {
        let s = input_b[t.z + x + t.w].x;
        u = (f32(x) + 0.5) / f32(w);
        if epi == 0u { v = ((f32(y) + 0.5) - s) / f32(h); }
        else { v = s_div(((f32(y) + ep) + 0.5) - s, f32(d.y)); }
    }
    return s_bilinear_or(0u, d, u, v, wrap_bits());
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.y * config.header.z + id.x;
    if id.x >= config.header.z || k >= config.header.y { return; }
    switch config.header.x {
        case 1u: { store(k, shear_pixel(k)); }
        case 2u: { store(k, unshear_pixel(k)); }
        case 3u: { store(k, s_resize_pixel(k)); }
        case 4u: { store(k, input_a[k]); }
        default: { store(k, point_pixel(k)); }
    }
}
