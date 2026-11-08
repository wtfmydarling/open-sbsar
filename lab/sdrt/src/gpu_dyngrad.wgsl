// Dynamic gradient (op 0x26): ops.rs "dyn_gradient" per pixel. input_a is the source (the
// node's size), input_b the gradient image (config.selectors: its descriptor), sampled at
// (t, coordinate), or (coordinate, t) when vertical, each mapped to texel centres
// (a * (n - 1) + 0.5) / n; t is the source's alpha (colour) or gray value, wrapped by
// rem_euclid(1) when repeating, else clamped to [0, 1].
// config.header.w bits: 0 repeat (also the sampler's tiling), 1 vertical, 2 the source is
// colour. config.values: the coordinate, gw - 1, gh - 1, 1/gw; config.extra.x: 1/gh (the
// gradient's sides are powers of two: exact reciprocals, ops.rs falls back otherwise).

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.y * config.header.z + id.x;
    if id.x >= config.header.z || k >= config.header.y { return; }
    let repeat = (config.header.w & 1u) != 0u;
    let p = input_a[k];
    var t = p.x;
    if (config.header.w & 4u) != 0u { t = p.w; }
    if repeat { t = rs_rem1(t); } else { t = rs_clamp(t, 0.0, 1.0); }
    t = barrier(t);
    var a = t;
    var b = config.values.x;
    if (config.header.w & 2u) != 0u {
        a = config.values.x;
        b = t;
    }
    let u = barrier(barrier(a * config.values.y) + 0.5) * config.values.w;
    let v = barrier(barrier(b * config.values.z) + 0.5) * config.extra.x;
    store(k, sample_b(u, v, repeat, repeat));
}
