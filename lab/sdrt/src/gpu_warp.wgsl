// Warp (op 0x0E) and directional warp (op 0x18), ops.rs "warp" / "directional_warp"; the
// helpers (s_*) are in gpu_sample.wgsl. ops.rs try_gpu_warp / try_gpu_dirwarp evaluate the
// node constants on the CPU exactly as the CPU functions do and pass them here.
//
// config.header.x selects the pass:
//   0 warp     - input_a the source (descriptor config.selectors), input_b the gradient
//                (descriptor bitcast(config.values)); parameters[0] = (kx, ky, 1/gw, 1/gh)
//   1 dirwarp  - input_a the source, input_b the intensity input at the node's size;
//                parameters[0] = (dx, dy, subtracted offset, 0)
//   2 resize   - image.rs "resized" of input_a (the directional warp's input_at)
// config.header.w bits: 0 wraps x, 1 wraps y, 2 the gradient has the node's size (texel
// differences, else bilinear differences at its own size), 3 the gradient / intensity image is
// colour (its alpha is read, else red).

fn wrap_bits() -> vec2<bool> {
    return vec2<bool>((config.header.w & 1u) != 0u, (config.header.w & 2u) != 0u);
}

// The gradient value the warp reads: alpha of a colour image, red of a gray one.
fn grad_value(p: vec4<f32>) -> f32 {
    if (config.header.w & 8u) != 0u { return p.w; }
    return p.x;
}

fn warp_pixel(k: u32) -> vec4<f32> {
    let w = config.header.z;
    let h = config.header.y / w;
    let x = k % w;
    let y = k / w;
    let wrap = wrap_bits();
    let gd = bitcast<vec4<u32>>(config.values);
    let par = config.parameters[0];
    // the node's sizes are powers of two: exact divisions
    let u0 = (f32(x) + 0.5) / f32(w);
    let v0 = (f32(y) + 0.5) / f32(h);
    var du: f32;
    var dv: f32;
    if (config.header.w & 4u) != 0u {
        // backward differences of the gradient texels
        let c = grad_value(s_fetch(1u, gd, i32(x), i32(y), wrap));
        du = par.x * (c - grad_value(s_fetch(1u, gd, i32(x) - 1, i32(y), wrap)));
        dv = par.y * (c - grad_value(s_fetch(1u, gd, i32(x), i32(y) - 1, wrap)));
    } else {
        // bilinear differences one gradient texel apart, at the gradient's own size
        let c = grad_value(s_bilinear_or(1u, gd, u0, v0, wrap));
        du = par.x * (c - grad_value(s_bilinear_or(1u, gd, u0 - par.z, v0, wrap)));
        dv = par.y * (c - grad_value(s_bilinear_or(1u, gd, u0, v0 - par.w, wrap)));
    }
    return s_bilinear_or(0u, config.selectors, u0 + barrier(du), v0 + barrier(dv), wrap);
}

fn dirwarp_pixel(k: u32) -> vec4<f32> {
    let w = config.header.z;
    let h = config.header.y / w;
    let x = k % w;
    let y = k / w;
    let wrap = wrap_bits();
    let ad = bitcast<vec4<u32>>(config.values);
    let par = config.parameters[0];
    let u0 = (f32(x) + 0.5) / f32(w);
    let v0 = (f32(y) + 0.5) / f32(h);
    var a: vec4<f32>;
    if (config.header.w & 4u) != 0u {
        a = s_fetch(1u, ad, i32(x), i32(y), wrap);
    } else {
        a = s_bilinear_or(1u, ad, u0, v0, wrap);
    }
    let m = grad_value(a) - par.z;
    let u = u0 + barrier(par.x * m);
    let v = v0 + barrier(par.y * m);
    return s_bilinear_or(0u, config.selectors, u, v, wrap);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.y * config.header.z + id.x;
    if id.x >= config.header.z || k >= config.header.y { return; }
    switch config.header.x {
        case 1u: { store(k, dirwarp_pixel(k)); }
        case 2u: { store(k, s_resize_pixel(k)); }
        default: { store(k, warp_pixel(k)); }
    }
}
