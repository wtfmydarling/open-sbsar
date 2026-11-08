// Emboss (op 0x10): ops.rs "emboss" per pixel. input_a is the source (the node's size),
// input_b the height (config.selectors: its descriptor). The height's backward differences are
// texel differences when it has the node's size (image.rs "fetch": tiled axes wrap, the others
// clamp), else three texture-unit samples at the node's pixel centres (sample_b). Then
//   s = (kx*gx)*cos + (ky*gy)*sin,  q = p + s * (s > 0 ? highlight : 1 - shadow)
// for R, G, B, alpha unchanged, every operation rounded as on the CPU.
// config.header.w bits: 0 wrap x, 1 wrap y, 2 the height has the node's size, 3 the height is
// colour (its alpha is read, else its gray value). config.values: kx, ky, cos, sin of the light
// angle (computed once on the CPU); config.extra.xy: 1/hw, 1/hh as the CPU computes them;
// config.parameters[0]: the highlight colour per channel, [1]: 1 - shadow per channel.

fn emboss_height(p: vec4<f32>) -> f32 {
    if (config.header.w & 8u) != 0u { return p.w; }
    return p.x;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = config.header.z;
    let k = id.y * w + id.x;
    if id.x >= w || k >= config.header.y { return; }
    let h = config.header.y / w;
    let wrap_x = (config.header.w & 1u) != 0u;
    let wrap_y = (config.header.w & 2u) != 0u;
    let x = i32(id.x);
    let y = i32(id.y);
    var gx: f32;
    var gy: f32;
    if (config.header.w & 4u) != 0u {
        let c = emboss_height(fetch_b(x, y, wrap_x, wrap_y));
        gx = barrier(c - emboss_height(fetch_b(x - 1, y, wrap_x, wrap_y)));
        gy = barrier(c - emboss_height(fetch_b(x, y - 1, wrap_x, wrap_y)));
    } else {
        // output sizes are powers of two: the divisions are exact
        let u = barrier(f32(x) + 0.5) / f32(w);
        let v = barrier(f32(y) + 0.5) / f32(h);
        let c = emboss_height(sample_b(u, v, wrap_x, wrap_y));
        gx = barrier(c - emboss_height(sample_b(barrier(u - config.extra.x), v, wrap_x, wrap_y)));
        gy = barrier(c - emboss_height(sample_b(u, barrier(v - config.extra.y), wrap_x, wrap_y)));
    }
    let s = barrier(barrier(barrier(config.values.x * gx) * config.values.z) + barrier(barrier(config.values.y * gy) * config.values.w));
    let p = input_a[k];
    var col = config.parameters[1];
    if s > 0.0 { col = config.parameters[0]; }
    let q = vec4<f32>(p.x + barrier(s * col.x), p.y + barrier(s * col.y), p.z + barrier(s * col.z), p.w);
    store(k, q);
}
