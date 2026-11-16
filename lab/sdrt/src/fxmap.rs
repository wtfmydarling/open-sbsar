//! FX-Map: the quadrant tree (research/sbsar/NOTES.md, "FX-Map", "Random numbers", "FX-Map
//! roughness"). Measured: seeds of quadrant children, iterations and switches; the roughness
//! weights; the shapes of the common patterns. Geometry (offsets, sizes, rotation) follows the
//! probes in lab/py; anything not measured is marked TODO.

use crate::image::Image;
use crate::ops::{QUAD_A, QUAD_C, Runtime};
use crate::program::{Node, Program};
use crate::vm::{Env, Eval, Func, Value};

const M24: u32 = 0xFF_FFFF;
/// Quadrant children: c0 = M·p + I, c(k+1) = M·c(k) + I, per depth of the child (1-based).
const CHILD: [(u32, u32); 12] = [
    (1691414, 52627),
    (5547014, 52673),
    (1289321, 52711),
    (10611682, 52747),
    (2603697, 52807),
    (13013189, 52859),
    (14482087, 52249),
    (4839483, 52289),
    (3906728, 52321),
    (4193611, 52379),
    (4656553, 52453),
    (3558434, 52511),
];
const ITER: (u32, u32) = (8532665, 14533);
/// Random streams of the functions inside an FX-Map tree: each kind of function draws from its own
/// LCG x ← A·x + C (mod 2²⁴), x₀ = the node's seed, rand = (x >> 1)/2²³ (lab/py/slotlcg.py,
/// itercount*.py). Quadrant parameters in record order: opacity, branch offset, frame offset,
/// pattern size, rotation, suppl.
const SLOT_LCG: [(u32, u32); 6] = [(QUAD_A, QUAD_C), (14154961, 50989), (1823937, 50707), (5812301, 50411), (3470325, 50131), (12382630, 50647)];
/// An Iterate's count function and a Switch's selector.
const ITER_COUNT_LCG: (u32, u32) = (9501597, 50591);
/// Seed-slot functions (quadrant, Iterate) use the node-level stream and keep all 24 state bits;
/// a quadrant's image-index function has its own stream (measured by the Codex helper,
/// cx/patterns/rng_stream_probe.py).
const SEED_SLOT_LCG: (u32, u32) = (16294893, 10493233);
const IMAGE_INDEX_LCG: (u32, u32) = (1664525, 7271263);
const SWITCH_LCG: (u32, u32) = (14464988, 50111);

/// The reference GPU's texture-derivative length, measured (scratch/fb/mipbisect.py: the mip level of a
/// rotated image stamp switches where this length, not the Euclidean one, crosses a half level):
/// length(a, b) = max·P(min/max), P from this table (r = min/max, P(r)), linear in between. Exact
/// would be √(1 + r²); the hardware's is about 1 + 0.342·r up to r ≈ 0.7, then steeper, 1.374 at r = 1.
const HW_LOD_P: [(f32, f32); 19] = [
    (0.00000, 1.00000), (0.04366, 1.01432), (0.08749, 1.02998), (0.13165, 1.04505), (0.17633, 1.06010), (0.22169, 1.07573), (0.26795, 1.09200), (0.31530, 1.10717), (0.36397, 1.12431), (0.41421, 1.14231), (0.46631, 1.16005), (0.52057, 1.17761), (0.57735, 1.19769), (0.63707, 1.21922), (0.70021, 1.23976), (0.76733, 1.26835), (0.83910, 1.30082), (0.91633, 1.33629), (1.00000, 1.37383),
];

/// Weight of the row length against the per-screen-axis length, by the rotation folded to 0–45°
/// (radians → weight), fitted to the 1.5:1, 2:1, 4:1 and 10:1 bisections (`scratch/fb/mipasp*.txt`):
/// 0 up to 27°, then ≈0.3 at 30°, 0.7 at 35°, 0.9 at 40°, 1 at 45°. Approximate: 1.1:1 stamps take
/// ≈0 at 30°, and 4:1 / 10:1 stamps at 45° take more than a level higher than this gives.
const ROW_BLEND: [(f32, f32); 7] = [(0.0, 0.0), (27.0, 0.0), (30.0, 0.3), (35.0, 0.7), (40.0, 0.89), (42.5, 0.94), (45.0, 1.0)];

fn row_blend(rad: f32) -> f32 {
    let d = rad.to_degrees().clamp(0.0, 45.0);
    let k = ROW_BLEND.iter().position(|&(x, _)| x >= d).unwrap_or(ROW_BLEND.len() - 1).max(1);
    let ((x0, w0), (x1, w1)) = (ROW_BLEND[k - 1], ROW_BLEND[k]);
    w0 + (w1 - w0) * (d - x0) / (x1 - x0)
}

fn hw_len(a: f32, b: f32) -> f32 {
    let (hi, lo) = (a.abs().max(b.abs()), a.abs().min(b.abs()));
    if hi == 0.0 {
        return 0.0;
    }
    let r = lo / hi;
    let k = HW_LOD_P.iter().position(|&(x, _)| x >= r).unwrap_or(HW_LOD_P.len() - 1).max(1);
    let ((x0, p0), (x1, p1)) = (HW_LOD_P[k - 1], HW_LOD_P[k]);
    hi * (p0 + (p1 - p0) * (r - x0) / (x1 - x0))
}

/// Exact half-turn matrices avoid moving nearest-filtered image coordinates across texel ties.
/// In multiwood 89, sin(-PI) in f32 selected a bright neighbouring row for half-turned stamps.
/// Both CPU sampling and the GPU's uploaded draw constants use this function. Coverage and
/// mip selection retain their independently measured rules.
fn sample_rotation(rot: f32, nearest_image: bool) -> (f32, f32) {
    // Do not generalize this to other cardinal angles or equivalent turns: the 14.1 Vulkan
    // interpolation ties at 0, quarter turns and 1.5 turns do not obey the same rule.
    if nearest_image && rot.abs() == 0.5 && crate::env_once!("SDRT_FXROT_LEGACY").is_err() {
        return (-1.0, 0.0);
    }
    let a = -rot * std::f32::consts::TAU;
    if crate::env_once!("SDRT_SBSROT").is_ok() {
        (crate::vm::sbs_cos(a), crate::vm::sbs_sin(a))
    } else {
        (a.cos(), a.sin())
    }
}

/// What a parameter function of a tree node sees.
struct QuadEnv<'a> {
    rt: &'a Runtime<'a>,
    /// The FX-Map's pattern inputs: its functions can sample them (ops 51/52).
    inputs: &'a [Vec<Image>],
    input_formats: &'a [u32],
    state: u32,
    number: f32,
    depth: f32,
    pos: [f32; 2],
    /// The LCG of this function's random stream (each kind of function has its own).
    lcg: (u32, u32),
    /// rand = state/2²⁴ (all 24 bits) instead of (state >> 1)/2²³.
    full_rand: bool,
    /// the last sampler's filter operand
    filter: u16,
    /// variable accesses, recorded for the parallel traversal
    track: Option<&'a std::cell::RefCell<VarTrack>>,
}

/// Variable slots a stretch of evaluations wrote, and read before writing them itself.
#[derive(Default, Clone)]
struct VarTrack {
    written: Vec<u16>,
    read_ext: Vec<u16>,
}

impl VarTrack {
    fn access(&mut self, slot: u16, write: bool) {
        if write {
            if !self.written.contains(&slot) {
                self.written.push(slot);
            }
        } else if !self.written.contains(&slot) && !self.read_ext.contains(&slot) {
            self.read_ext.push(slot);
        }
    }
}

impl Env for QuadEnv<'_> {
    fn input(&self, uid: u32) -> Value {
        self.rt.input(uid)
    }
    fn sysvar(&self, index: u16) -> Value {
        if let Some(v) = crate::ops::size_sysvar(index, self.rt.cur_size) {
            return v;
        }
        match index {
            8 => Value::float(&self.pos),
            10 => Value::f1(self.number),
            11 => Value::f1(self.depth.exp2()),
            12 => Value::f1(self.depth),
            _ => Value::f1(0.0),
        }
    }
    fn sample(&mut self, input: usize, pos: [f32; 2], color: bool, nearest: bool) -> [f32; 4] {
        // an FX-Map function sampling a pattern input (officeceilingsubstance001 node 401): as in a
        // pixel processor, nearest unless the filter operand is 1, tiling on both axes
        let Some(img) = self.inputs.get(input).and_then(|c| c.first()) else { return [0.0; 4] };
        // functions run on the CPU: exact float bilinear weights, not the texture unit's
        // (scratch/fb/fxsamp.py: 18/18 bit-exact; 1/256 weights up to 6e-4, UNORM integers 2e-3).
        // SDRT_FXSAMPLE=256 | tex for the other rules
        let mode = crate::env_once!("SDRT_FXSAMPLE");
        if crate::env_once!("SDRT_FXSDBG").is_ok() {
            let (x, y) = (pos[0] * img.w as f32 - 0.5, pos[1] * img.h as f32 - 0.5);
            eprintln!("fxsample in {} {}x{} near {} pos ({}, {}) frac ({:.5}, {:.5}) exact {:?} q256 {:?}", input, img.w, img.h, nearest, pos[0], pos[1], x - x.floor(), y - y.floor(), img.bilinear_exact(pos[0], pos[1], (true, true))[0], img.bilinear(pos[0], pos[1], (true, true))[0]);
        }
        // filter operand bit 1: edges clamped (see vm.rs op 51/52)
        let wr = self.filter & 2 == 0;
        let s = if nearest {
            img.nearest(pos[0], pos[1], (wr, wr))
        } else if mode.as_deref() == Ok("256") {
            img.bilinear(pos[0], pos[1], (wr, wr))
        } else if mode.as_deref() == Ok("tex") {
            img.bilinear_or(pos[0], pos[1], (wr, wr), None, self.input_formats.get(input).copied().unwrap_or(3))
        } else {
            img.bilinear_exact(pos[0], pos[1], (wr, wr))
        };
        if color || !img.color {
            s
        } else {
            let l = (s[0] + s[1] + s[2]) / 3.0;
            [l, l, l, 1.0]
        }
    }
    fn vp_value(&self, node: usize) -> Value {
        self.rt.vp_slot(node)
    }
    fn sample_filter(&mut self, filter: u16) {
        self.filter = filter;
    }
    fn var_access(&mut self, slot: u16, write: bool) {
        if let Some(t) = self.track {
            t.borrow_mut().access(slot, write);
        }
    }
    fn rand(&mut self) -> f32 {
        self.state = (self.lcg.0.wrapping_mul(self.state).wrapping_add(self.lcg.1)) & M24;
        if self.full_rand { self.state as f32 / 16777216.0 } else { (self.state >> 1) as f32 / 8388608.0 }
    }
    fn cpu_trig(&self) -> bool {
        crate::env_once!("SDRT_LIBMTRIG").is_err()
    }
}

/// Generated patterns at local coordinates u, v in [-1, 1] (measured: lab/py/patprobe.py).
/// Is a pixel centre outside a pattern's square? Local coordinates in (−1, 1]: a centre exactly on
/// the boundary of two cells belongs to the cell left of / above it (metalsubstance007 node 0, a
/// 5-level quadtree on 16 px, is exact only so).
fn outside(lx: f32, ly: f32) -> bool {
    !(lx > -1.0 && lx <= 1.0 && ly > -1.0 && ly <= 1.0)
}

/// A pattern's footprint in f64: rotation and half sizes.
struct Cover {
    ca: f64,
    sa: f64,
    hx: f64,
    hy: f64,
    /// half extents of the screen-space bounding box
    ex: f64,
    ey: f64,
    /// test switches, read once per draw: SDRT_COVER32 (f32 coverage), SDRT_COVEROLD (local
    /// (−1, 1] rule), SDRT_LX64 (f64 local coordinates)
    f32_cover: bool,
    old_cover: bool,
    lx64: bool,
    /// SDRT_SNAP=bits: GPU-style coverage test instead (test switch): the quad's corners in pixel
    /// units snapped to 1/2^bits, edge functions at pixel centres, ties to right/bottom edges
    snap: Option<u32>,
    /// SDRT_SNAPLX=bits[:diagonal]: local coordinates interpolated over the snapped quad's two
    /// triangles (diagonal 0: corners 0–2, 1: corners 1–3), as a GPU interpolates attributes (test)
    snap_lx: Option<(u32, u8)>,
}

/// A quad rasterized as by a GPU: fixed-point corners, edge functions.
struct Snap {
    v: [[i64; 2]; 4],
    s: i64,
    one: i64,
    /// tie rule, SDRT_SNAPTIE=xy (x, y each + or −; default ++: right, then bottom edges)
    tie: (i64, i64),
}

impl Snap {
    /// Local coordinates at pixel (px, py): the corners' (±1, ±1) interpolated over the triangle
    /// of the quad that holds the pixel centre.
    fn local(&self, px: i64, py: i64, diag: u8) -> (f32, f32) {
        const A: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let p = [(px * self.one + self.one / 2) as f64, (py * self.one + self.one / 2) as f64];
        let v = self.v.map(|q| [q[0] as f64, q[1] as f64]);
        let tris: [[usize; 3]; 2] = if diag == 0 { [[0, 1, 2], [0, 2, 3]] } else { [[0, 1, 3], [1, 2, 3]] };
        let mut best = None;
        for t in tris {
            let (a, b, c) = (v[t[0]], v[t[1]], v[t[2]]);
            let d = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
            if d == 0.0 {
                continue;
            }
            let w1 = ((p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])) / d;
            let w2 = ((b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])) / d;
            let w0 = 1.0 - w1 - w2;
            let worst = w0.min(w1).min(w2);
            if best.as_ref().is_none_or(|(m, _, _)| worst > *m) {
                best = Some((worst, t, [w0, w1, w2]));
            }
        }
        let Some((_, t, w)) = best else { return (2.0, 2.0) };
        let at = |j: usize| (0..3).map(|i| w[i] * A[t[i]][j]).sum::<f64>() as f32;
        (at(0), at(1))
    }

    fn inside(&self, px: i64, py: i64) -> bool {
        let p = [px * self.one + self.one / 2, py * self.one + self.one / 2];
        for i in 0..4 {
            let (a, b) = (self.v[i], self.v[(i + 1) % 4]);
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let cr = self.s as i128 * (ex as i128 * (p[1] - a[1]) as i128 - ey as i128 * (p[0] - a[0]) as i128);
            if cr < 0 {
                return false;
            }
            if cr == 0 {
                // outward normal s·(ey, −ex): a tie belongs to an edge whose normal points along
                // tie.0 in x, or along tie.1 in y for a horizontal edge
                let (nx, ny) = (self.s * ey, -self.s * ex);
                if !(nx * self.tie.0 > 0 || (nx == 0 && ny * self.tie.1 > 0)) {
                    return false;
                }
            }
        }
        true
    }
}

impl Cover {
    fn new(rot: f32, sz: [f32; 2]) -> Cover {
        let a = -(rot as f64) * std::f64::consts::TAU;
        let (ca, sa) = if crate::env_once!("SDRT_COVERSBS").is_ok() {
            // the rotation as Substance's CPU sine computes it (the matrix is built on the CPU)
            if crate::env_once!("SDRT_COVERSBS").as_deref() == Ok("pos") {
                (crate::vm::sbs_trig_turns(rot, 0.5) as f64, -(crate::vm::sbs_sin_turns(rot) as f64))
            } else {
                (crate::vm::sbs_trig_turns(-rot, 0.5) as f64, crate::vm::sbs_sin_turns(-rot) as f64)
            }
        } else { (a.cos(), a.sin()) };
        let (hx, hy) = (sz[0] as f64 / 2.0, sz[1] as f64 / 2.0);
        Cover { ca, sa, hx, hy, ex: (ca * hx).abs() + (sa * hy).abs(), ey: (sa * hx).abs() + (ca * hy).abs(),
            f32_cover: crate::env_once!("SDRT_COVER32").is_ok(), old_cover: crate::env_once!("SDRT_COVEROLD").is_ok(), lx64: crate::env_once!("SDRT_LX64").is_ok(),
            snap: crate::env_once!("SDRT_SNAP").ok().and_then(|x| x.parse().ok()),
            snap_lx: crate::env_once!("SDRT_SNAPLX").ok().and_then(|x| {
                let mut it = x.split(':');
                Some((it.next()?.parse().ok()?, it.next().and_then(|d| d.parse().ok()).unwrap_or(0)))
            }) }
    }
    /// The snapped quad of a pattern centred at (cx, cy), in pixel units: pixel = (uv − o)·k.
    fn quad(&self, cx: f64, cy: f64, o: [f64; 2], k: [f64; 2]) -> Option<Snap> {
        self.quad_bits(cx, cy, o, k, self.snap?)
    }
    /// The quad for interpolated local coordinates (SDRT_SNAPLX), if that test switch is on.
    fn lx_quad(&self, cx: f64, cy: f64, o: [f64; 2], k: [f64; 2]) -> Option<Snap> {
        self.quad_bits(cx, cy, o, k, self.snap_lx?.0)
    }
    fn quad_bits(&self, cx: f64, cy: f64, o: [f64; 2], k: [f64; 2], bits: u32) -> Option<Snap> {
        let one = 1i64 << bits;
        let mut v = [[0i64; 2]; 4];
        for (j, (u, w)) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].into_iter().enumerate() {
            let (lx, ly) = (u * self.hx, w * self.hy);
            let (dx, dy) = (self.ca * lx + self.sa * ly, -self.sa * lx + self.ca * ly);
            v[j] = [((cx + dx - o[0]) * k[0] * one as f64).round() as i64, ((cy + dy - o[1]) * k[1] * one as f64).round() as i64];
        }
        let mut area = 0i128;
        for i in 0..4 {
            let (a, b) = (v[i], v[(i + 1) % 4]);
            area += a[0] as i128 * b[1] as i128 - b[0] as i128 * a[1] as i128;
        }
        let t = crate::env_once!("SDRT_SNAPTIE").unwrap_or_else(|_| "++".into());
        let sg = |c: Option<char>| if c == Some('-') { -1 } else { 1 };
        Some(Snap { v, s: if area >= 0 { 1 } else { -1 }, one, tie: (sg(t.chars().next()), sg(t.chars().nth(1))) })
    }
    /// local coordinates in f64, rounded to f32
    fn local(&self, dx: f64, dy: f64) -> (f32, f32) {
        (((self.ca * dx - self.sa * dy) / self.hx) as f32, ((self.sa * dx + self.ca * dy) / self.hy) as f32)
    }
}

fn outside64(c: &Cover, dx: f64, dy: f64) -> bool {
    if c.f32_cover {
        let (lx, ly) = c.local(dx, dy);
        return outside(lx, ly);
    }
    let lx = (c.ca * dx - c.sa * dy) / c.hx;
    let ly = (c.sa * dx + c.ca * dy) / c.hy;
    if c.old_cover {
        return !(lx > -1.0 && lx <= 1.0 && ly > -1.0 && ly <= 1.0);
    }
    // The stamp is drawn as its screen-space bounding box: a pixel centre on the box's left or top
    // edge is outside, on its right or bottom edge inside; within the box the pattern's own edges
    // are inclusive. Ties (centres on an edge) are taken within 1e-6 of the extent: f32 inputs
    // (terrazzosubstance001 node 921: size 0.70710677 = √2/2 − 2e-8 relative) and f64 rounding
    // (scratch/fb/tieprobe.py: axis-aligned squares and rectangles at any quarter turn, 45°
    // diamonds with all four edges through centres).
    const T: f64 = 1e-6;
    let (bx, by) = (dx / c.ex, dy / c.ey);
    !(bx > -1.0 + T && bx <= 1.0 + T && by > -1.0 + T && by <= 1.0 + T && lx.abs() <= 1.0 + T && ly.abs() <= 1.0 + T)
}

fn pattern(kind: u32, u: f32, v: f32, suppl: f32) -> f32 {
    let r2 = u * u + v * v;
    let r = r2.sqrt();
    match kind {
        0 => 1.0,                                    // square
        1 => (r <= 1.0) as i32 as f32,               // disc
        2 => (1.0 - r2).max(0.0),                    // paraboloid
        3 => (1.0 - r2).max(0.0).powi(4),            // bell
        // gaussian: e^(−11.1 r²) shifted to reach 0 at r = 1 (cx/patterns/out/fresh base32, max 3.4e-7)
        4 => ((-11.1 * r2).exp() - (-11.1f32).exp()).max(0.0),
        5 => (1.0 - r).max(0.0).powi(4),             // thorn
        6 => 1.0 - u.abs().max(v.abs()),             // pyramid
        7 => {
            // brick: patternsuppl = bevel width; smoothstep of the distance to each edge, both axes
            // multiplied (lab/py/brickprobe.py)
            if suppl <= 0.0 {
                1.0
            } else {
                let sm = |d: f32| {
                    let t = (d / suppl).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                };
                sm(1.0 - u.abs()) * sm(1.0 - v.abs())
            }
        }
        8 => {
            // Gradation fades out linearly for a positive supplement; negative
            // supplements are ignored (fresh 32f supplement sweep).
            (u + 1.0) * 0.5 * (1.0 - suppl.max(0.0))
        }
        9 => {
            // Waves: three lobes at zero, with twelve additional lobes per
            // positive supplement.  Its carrier is *not* hardware cos: it is
            // the folded quartic used by the generated-pattern evaluator.
            // Let x measure a quarter-wave (0 at an extremum, 1 at a zero).
            // The measured coefficient is 1.223936 (cos_path_probe.py).
            let waves = 3.0 + 12.0 * suppl.max(0.0);
            let cycle = (0.5 * waves * u).rem_euclid(1.0);
            let half = cycle.min(1.0 - cycle);
            let x = 4.0 * half.min(0.5 - half);
            let a = 1.223_936_f32;
            let wave = 1.0 - a * x * x + (a - 1.0) * x * x * x * x;
            let carrier = if half < 0.25 { wave } else { -wave };
            (1.0 - r2).max(0.0) * (carrier + 1.0) * 0.5
        }
        10 => {
            // The half-bell is a bell multiplied by a supplement-controlled
            // smooth transition across x.  At s=0 this is a sharp left half;
            // above one the transition folds around from the left.
            let d = 1.0 - suppl.max(0.0);
            let smooth = |x: f32| {
                let x = x.clamp(0.0, 1.0);
                x * x * (3.0 - 2.0 * x)
            };
            let factor = if d > 0.0 {
                1.0 - smooth(u / d)
            } else if d < 0.0 {
                smooth((u - d) / -d)
            } else {
                (u <= 0.0) as i32 as f32
            };
            (1.0 - r2).max(0.0).powi(4) * factor
        }
        11 => {
            // Ridged bell: a bell with an independent squared horizontal
            // taper.  Its supplement slot has no effect.
            (1.0 - r2).max(0.0).powi(4) * (1.0 - u.abs()).powi(2)
        }
        12 => {
            // Crescent.  s<=0 is the default left-to-right smooth mask.  At
            // s=1 the denominator is singular and Adobe returns zero; past it
            // the (negative) smoothstep argument clamps to zero, revealing the
            // unmasked radial base.
            let d = 1.0 - suppl.max(0.0);
            let smooth = |x: f32| {
                let x = x.clamp(0.0, 1.0);
                x * x * (3.0 - 2.0 * x)
            };
            let factor = if d == 0.0 { 0.0 } else { 1.0 - smooth((u + 1.0) / (2.0 * d)) };
            6.75 * r2 * (1.0 - r2).max(0.0).powi(2) * factor
        }
        13 => {
            // Capsule: a quadratic distance field.  Positive supplement
            // lengthens its flat middle by 0.77*s; negative values are ignored.
            let a = (u.abs() - 0.77 * suppl.max(0.0)).max(0.0);
            (1.0 - 20.0 * (a * a + v * v)).max(0.0)
        }
        14 => (1.0 - r).max(0.0),                    // cone
        15 => (u + 1.0) * 0.5 + suppl,               // gradation offset (unclamped)
        _ => (1.0 - r2).max(0.0).powi(4),
    }
}

const GPU_DRAW_LIMIT: usize = 2_097_152;

#[allow(clippy::type_complexity)]
/// A recorded draw: src, index, image flags, centre, size, rotation, suppl, opacity, max blend,
/// weight.
type DrawRec = (u32, u32, u32, [f32; 2], [f32; 2], f32, f32, V4, bool, f32);

/// The parallel traversal's record: a draw, or a subtree (node offset, state, the index of the
/// variables at that point of the walk in `Ctx::snaps`) evaluated apart.
enum Item {
    D(DrawRec),
    T(usize, State, u32),
}

struct Ctx<'a> {
    gpu_collect: bool,
    deferred: Option<Vec<DrawRec>>,
    /// Parallel traversal (see `render`): draws and subtrees past `split_at` node calls are
    /// recorded here instead of drawn / entered.
    collect: Option<Vec<Item>>,
    /// distinct variable states at the recorded subtrees (consecutive equal ones shared)
    snaps: Vec<Vec<(u16, Value)>>,
    /// speculative workers: a shared stop flag and this worker's work in the current window (node
    /// calls). A task that reads a value an earlier task would have set can run away (a count read
    /// from a variable: metalplatessubstance002 node 31, > 20 GB) before the check after it —
    /// past the budget everything stops and the node runs sequentially.
    spec: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    work: u64,
    /// tree nodes visited by this context (runaway guard)
    visits: u64,
    /// variable entries the snapshots may still hold; exhausted = the walk stops, sequential
    /// (metalplatessubstance002: a counter changed at every copy → one full snapshot per task, > 20 GB)
    snap_budget: usize,
    /// variable accesses: the top walk's after its first subtree, or a worker's per subtree
    track: Option<std::cell::RefCell<VarTrack>>,
    call_depth: u32,
    split_at: u32,
    /// no SDRT_SNAP / SDRT_SNAPLX test switch (the early out in draw assumes the plain coverage)
    no_snap: bool,
    rt: &'a Runtime<'a>,
    p: &'a Program,
    out: Image,
    color: bool,
    /// 8-bit color targets quantize the premultiplied fragment before blending, and the target
    /// after each layer. Higher precision formats use a different accumulation path.
    quantized_color: bool,
    quantized_gray: bool,
    max_depth: u32,
    hurst: f32,
    /// The global opacity (opacitymult): scales every quadrant's opacity (lab/py/opmultprobe.py).
    opmult: f32,
    /// Render and tiling ROIs (x0, x1, y0, y1) in FX-Map space, when the node has a render ROI.
    roi: Option<([f32; 4], [f32; 4])>,
    wrap: (bool, bool),
    /// Pattern inputs as mip chains (level 0 = the image).
    inputs: std::sync::Arc<Vec<Vec<Image>>>,
    input_formats: Vec<u32>,
    funcs: std::cell::RefCell<FxMap<usize, std::sync::Arc<Func>>>,
    /// weight(depth) for depths 0..64
    weights: Vec<f32>,
    /// The VM's memo buffers, reused by every evaluation of this render.
    bufs: std::cell::RefCell<crate::vm::EvalBufs>,
    /// Function variables shared by every evaluation of one render (set in one, read in later ones).
    vars: std::cell::RefCell<Vec<(u16, Value)>>,
    unsupported: Vec<String>,
}

/// A parameter value of up to 4 floats (no allocation per quadrant); derefs to its slice.
#[derive(Clone, Copy)]
struct V4 {
    n: u8,
    a: [f32; 4],
}

impl V4 {
    fn of(v: &[f32]) -> V4 {
        let mut a = [0.0; 4];
        let n = v.len().min(4);
        a[..n].copy_from_slice(&v[..n]);
        V4 { n: n as u8, a }
    }
}

impl std::ops::Deref for V4 {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.a[..self.n as usize]
    }
}

impl std::fmt::Debug for V4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

/// FxHash (rustc's): the function cache is looked up once per evaluation; SipHash cost ~9%.
#[derive(Default)]
struct FxHasher(u64);

impl std::hash::Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_usize(&mut self, x: usize) {
        self.write_u64(x as u64);
    }
}

type FxMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<FxHasher>>;

#[derive(Clone, Copy)]
struct State {
    depth: u32,
    center: [f32; 2],
    seed: u32,
    branch: [f32; 2],
    number: f32,
}

impl<'a> Ctx<'a> {
    fn eval(&self, off: usize, st: &State) -> Value {
        self.eval_with(off, st, (QUAD_A, QUAD_C))
    }

    fn eval_with(&self, off: usize, st: &State, lcg: (u32, u32)) -> Value {
        self.eval_mode(off, st, lcg, false)
    }

    fn eval_mode(&self, off: usize, st: &State, lcg: (u32, u32), full_rand: bool) -> Value {
        let t0 = if prof() { Some(std::time::Instant::now()) } else { None };
        let f = self.funcs.borrow_mut().entry(off).or_insert_with(|| {
            if self.rt.incr.is_some() {
                self.rt.fx_trace.lock().unwrap_or_else(|e| e.into_inner()).push(off);
            }
            std::sync::Arc::new(Func::decode(self.p, off))
        }).clone();
        let mut env = QuadEnv { rt: self.rt, inputs: &self.inputs, input_formats: &self.input_formats, state: st.seed & M24, number: st.number, depth: st.depth as f32, pos: st.center, lcg, full_rand, filter: 0, track: self.track.as_ref() };
        let v = Eval::run_with_vars_in(&f, &mut env, &mut self.vars.borrow_mut(), &mut self.bufs.borrow_mut());
        if let Some(t0) = t0 {
            if crate::env_once!("SDRT_FXPROF").as_deref() == Ok("2") {
                let mut m = PROF_FN.lock().unwrap();
                let e = m.entry(off).or_insert((0, f.ins.len(), f.pure()));
                e.0 += 1;
            }
            PROF_EVAL_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            PROF_EVALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            PROF_INS.fetch_add(f.ins.len() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        v
    }

    /// A 2-bit slot inside a parameter block: (value, next offset).
    /// A seed slot: the constant word, or a function's integer result (a float result is
    /// truncated), and the next offset.
    fn seed_slot(&self, at: usize, bits: u32, st: &State) -> (u32, usize) {
        match bits & 3 {
            1 => (self.p.w(at), at + 4),
            2 => {
                let v = self.eval_mode(self.p.w(at) as usize, st, SEED_SLOT_LCG, true);
                // an integer result becomes the seed as the bit pattern of its float value
                // (scratch/fb/varprobe.py on challenge_rooftop node 0: var0 = 125089088 computed by
                // a cast, and var0 = 0, 1, 5, 12345, 40000, 65541, 16777223, 77777777 from aligned
                // constants — all exact). Float results: unmeasured.
                let k = match v {
                    Value::I(_, x) => (x[0] as f32).to_bits(),
                    _ => v.as_f() as i64 as u32,
                };
                (k, at + 4)
            }
            _ => (0, at),
        }
    }

    #[allow(dead_code)]
    fn slot(&self, at: usize, bits: u32, n: usize, st: &State) -> (Option<V4>, usize) {
        self.slot_with(at, bits, n, st, (QUAD_A, QUAD_C))
    }

    fn slot_with(&self, at: usize, bits: u32, n: usize, st: &State, lcg: (u32, u32)) -> (Option<V4>, usize) {
        match bits & 3 {
            1 => {
                let mut a = [0.0; 4];
                for k in 0..n.min(4) {
                    a[k] = self.p.f(at + 4 * k);
                }
                (Some(V4 { n: n.min(4) as u8, a }), at + 4 * n)
            }
            2 => {
                let v = self.eval_with(self.p.w(at) as usize, st, lcg);
                (Some(V4::of(&v.fv()[..v.n().max(1)])), at + 4)
            }
            _ => (None, at),
        }
    }

    fn weight(&self, depth: u32) -> f32 {
        // per-depth table (the same expressions, computed once per render)
        if let Some(&w) = self.weights.get(depth as usize) {
            return w;
        }
        self.weight_calc(depth)
    }

    /// A context for evaluating subtrees apart: same node, own function cache, buffers and
    /// variables (a copy), draws recorded.
    fn worker(&self) -> Ctx<'a> {
        Ctx {
            gpu_collect: self.gpu_collect,
            deferred: Some(vec![]),
            collect: None,
            snaps: vec![],
            snap_budget: 0,
            spec: None,
            work: 0,
            visits: 0,
            track: Some(Default::default()),
            call_depth: 0,
            split_at: u32::MAX,
            no_snap: self.no_snap,
            rt: self.rt,
            p: self.p,
            out: Image::new(1, 1, self.color, [0.0; 4]),
            color: self.color,
            quantized_color: self.quantized_color,
            quantized_gray: self.quantized_gray,
            max_depth: self.max_depth,
            hurst: self.hurst,
            opmult: self.opmult,
            roi: self.roi,
            wrap: self.wrap,
            inputs: self.inputs.clone(),
            input_formats: self.input_formats.clone(),
            funcs: Default::default(),
            weights: self.weights.clone(),
            bufs: Default::default(),
            vars: std::cell::RefCell::new(self.vars.borrow().clone()),
            unsupported: vec![],
        }
    }

    fn weight_calc(&self, depth: u32) -> f32 {
        let h = self.hurst;
        if h == 0.0 {
            return 1.0;
        }
        let sum: f32 = (0..=self.max_depth).map(|k| h.powi(k as i32)).sum();
        h.powi(depth as i32) / sum
    }

    /// Upper bound on the number of draws a tree can produce, evaluated without allocating
    /// DrawRecs or traversing identical subtree copies. Used to decide before collecting whether
    /// a node will exceed the GPU draw budget and should fall back immediately to CPU.
    fn estimate_draws(&self, off: usize, st: State, limit: usize) -> usize {
        if off == 0 || st.depth > 64 {
            return 0;
        }
        let p = self.p;
        let w = p.w(off);
        let kind = w & 7;
        let mut o = off + 4;
        match kind {
            0 => {
                let src = (w >> 6) & 3;
                let mut count = if src != 0 { 1usize } else { 0usize };
                let ifl = if w & 0x8000_0000 != 0 { o += 4; p.w(off + 4) } else { 0 };
                if (w & 0x30) != 0 || [19, 21, 23, 25, 27, 29].iter().any(|b| (w >> b) & 3 != 0) || ifl & 2 != 0 {
                    o += 4;
                }
                let depth = st.depth + 1;
        if depth >= 64 {
            crate::cap_hit("fx depth");
            return count;
        }
                if w & 0x10000 != 0 {
                    for k in 0..4 {
                        let child = p.w(o + 4 * k) as usize;
                        if child != 0 {
                            count = count.saturating_add(self.estimate_draws(child, State { depth, ..st }, limit));
                            if count >= limit {
                                return limit;
                            }
                        }
                    }
                } else if w & 0x20000 != 0 {
                    let child = p.w(o) as usize;
                    if child != 0 {
                        let c = self.estimate_draws(child, State { depth, ..st }, limit);
                        count = count.saturating_add(c.saturating_mul(4));
                    }
                }
                count
            }
            1 => {
                let _parent = st;
                let mut st = st;
                if (w >> 4) & 3 != 0 {
                    let (k, n) = self.seed_slot(o, (w >> 4) & 3, &st);
                    st.seed = if (w >> 3) & 1 == 0 { k } else { st.seed.wrapping_add(k) } & M24;
                    o = n;
                }
                if w & 0x80 != 0 {
                    o += 4;
                    let c0 = p.w(o) as usize;
                    let c1 = p.w(o + 4) as usize;
                    let e0 = if c0 != 0 { self.estimate_draws(c0, st, limit) } else { 0 };
                    let e1 = if c1 != 0 { self.estimate_draws(c1, st, limit) } else { 0 };
                    e0.max(e1)
                } else {
                    let value = (w >> 6) & 1;
                    let child = p.w(o + 4 * value as usize) as usize;
                    if child != 0 {
                        self.estimate_draws(child, st, limit)
                    } else {
                        0
                    }
                }
            }
            3 => {
                let parent = st;
                let mut st = st;
                if (w >> 4) & 3 != 0 {
                    let (k, n) = self.seed_slot(o, (w >> 4) & 3, &st);
                    st.seed = if (w >> 3) & 1 == 0 { k } else { st.seed.wrapping_add(k) } & M24;
                    o = n;
                }
                let mut count = (w >> 8) as usize;
                if w & 0x80 != 0 {
                    let fst = if crate::env_once!("SDRT_CNTPARENT").is_ok() { parent } else { st };
                    let vars_snap = self.vars.borrow().clone();
                    count = self.eval_with(p.w(o) as usize, &fst, ITER_COUNT_LCG).as_f().max(0.0) as usize;
                    *self.vars.borrow_mut() = vars_snap;
                    o += 4;
                }
                let c1 = p.w(o) as usize;
                let c0 = if w & 0x40 != 0 && !crate::env_once!("SDRT_OUT0LATE").is_ok() {
                    p.w(o + 4) as usize
                } else {
                    0
                };
                let mut total = if c0 != 0 {
                    self.estimate_draws(c0, st, limit)
                } else {
                    0
                };
                if c1 != 0 && count > 0 {
                    let child_draws = self.estimate_draws(c1, st, limit);
                    total = total.saturating_add(child_draws.saturating_mul(count));
                }
                total
            }
            _ => 0,
        }
    }

    fn node(&mut self, off: usize, st: State) {
        if st.depth >= 64 || self.call_depth >= 64 {
            crate::cap_hit("fx node depth");
            return;
        }
        // a runaway (malformed) tree fails the render; real trees stay far below this
        // (castle_wall 805: 17M stamps)
        self.visits += 1;
        if self.visits > 1 << 28 {
            crate::cap_hit("fx visits");
            panic!("FX-Map: more than 2^28 tree nodes visited (malformed program?)");
        }
        if self.gpu_collect && self.deferred.as_ref().is_some_and(|d| d.len() >= GPU_DRAW_LIMIT) {
            return;
        }
        if self.collect.is_some() && self.snap_budget == 0 {
            return;
        }
        if let Some(a) = &self.spec {
            if a.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            let draws = self.deferred.as_ref().map_or(0, |d| d.len());
            self.work += 1;
            if self.work > 50_000_000 || draws > 20_000_000 {
                a.store(true, std::sync::atomic::Ordering::Relaxed);
                return;
            }
        }
        if let Some(c) = self.collect.as_mut() {
            if self.call_depth >= self.split_at {
                let first = self.snaps.is_empty();
                if self.snaps.last().is_none_or(|v| *v != *self.vars.borrow()) {
                    let n = self.vars.borrow().len().max(1);
                    if n > self.snap_budget {
                        self.snap_budget = 0;
                        return;
                    }
                    self.snap_budget -= n;
                    self.snaps.push(self.vars.borrow().clone());
                }
                c.push(Item::T(off, st, (self.snaps.len() - 1) as u32));
                if first {
                    // from here on, every variable the top walk touches counts
                    self.track = Some(Default::default());
                }
                return;
            }
        }
        self.call_depth += 1;
        self.node_body(off, st);
        self.call_depth -= 1;
    }

    fn node_body(&mut self, off: usize, st: State) {
        let p = self.p;
        let w = p.w(off);
        let kind = w & 7;
        let mut o = off + 4;
        match kind {
            0 => self.quadrant(off, w, st),
            1 | 3 => {
                let parent = st;
                let mut st = st;
                if (w >> 4) & 3 != 0 {
                    let (k, n) = self.seed_slot(o, (w >> 4) & 3, &st);
                    // a seed slot replaces the seed, or with the inherit bit is added to the
                    // parent's (lab/py/seedslot.py)
                    st.seed = if (w >> 3) & 1 == 0 { k } else { st.seed.wrapping_add(k) } & M24;
                    o = n;
                }
                if kind == 1 {
                    let mut value = (w >> 6) & 1;
                    if w & 0x80 != 0 {
                        let fst = if crate::env_once!("SDRT_SWPARENT").is_ok() { parent } else { st };
                        value = self.eval_with(p.w(o) as usize, &fst, SWITCH_LCG).as_bool() as u32;
                        if crate::env_once!("SDRT_FXLOG").is_ok() {
                            let f = Func::decode(self.p, p.w(o) as usize);
                            eprintln!("fxswitch fn {} value {} ins {:?}", p.w(o), value, f.ins.iter().map(|i| (i.op, i.ty, i.ops.clone())).collect::<Vec<_>>());
                        }
                        o += 4;
                    }
                    let child = p.w(o + 4 * value as usize) as usize;
                    if child != 0 {
                        self.node(child, st);
                    }
                } else {
                    // the count fills the 24 bits above bit 8 (cartoon_slab node 1745: 250,000 = 500 × 500)
                    let mut count = w >> 8;
                    
                    if w & 0x80 != 0 {
                        // the count function draws from its own stream: ITER_COUNT applied to the
                        // Iterate's seed (lab/py/itercount*.py)
                        let fst = if crate::env_once!("SDRT_CNTPARENT").is_ok() { parent } else { st };
                        count = self.eval_with(p.w(o) as usize, &fst, ITER_COUNT_LCG).as_f().max(0.0) as u32;
                        // a speculative task with a large count that may come from a variable an
                        // earlier task sets: stop speculating (see `spec`)
                        if count > 65_536 {
                            if let (Some(a), Some(t)) = (&self.spec, &self.track) {
                                if !t.borrow().read_ext.is_empty() {
                                    a.store(true, std::sync::atomic::Ordering::Relaxed);
                                    return;
                                }
                            }
                        }
                        if crate::env_once!("SDRT_FXLOG").is_ok() {
                            eprintln!("fxiter fn {} count {}", p.w(o), count);
                        }
                        o += 4;
                    }
                    let c1 = p.w(o) as usize;
                    // output0 (when connected) runs once with the Iterate's own seed, before output1 (iterorder.py)
                    let late0 = crate::env_once!("SDRT_OUT0LATE").is_ok();
                    if w & 0x40 != 0 && !late0 {
                        let c0 = p.w(o + 4) as usize;
                        if c0 != 0 {
                            self.node(c0, st);
                        }
                    }
                    let mut s = st.seed;
                    if crate::env_once!("SDRT_ITERREV").is_ok() {
                        // test: the copies drawn last to first (same seeds and numbers)
                        let mut seeds = vec![];
                        for _ in 0..count { s = (ITER.0.wrapping_mul(s).wrapping_add(ITER.1)) & M24; seeds.push(s); }
                        for j in (0..count).rev() {
                            if c1 != 0 { self.node(c1, State { seed: seeds[j as usize], number: j as f32, ..st }); }
                        }
                    } else {
                    // SDRT_FXNUM (test): 1 = numbers from 1, flat = parent's number × count + j
                    let numode = crate::env_once!("SDRT_FXNUM").unwrap_or("");
                    for j in 0..count {
                        s = (ITER.0.wrapping_mul(s).wrapping_add(ITER.1)) & M24;
                        if c1 != 0 {
                            let number = match numode { "1" => j as f32 + 1.0, "flat" => st.number * count as f32 + j as f32, _ => j as f32 };
                            self.node(c1, State { seed: s, number, ..st });
                        }
                    }
                    }
                    if w & 0x40 != 0 && late0 {
                        let c0 = p.w(o + 4) as usize;
                        if c0 != 0 {
                            self.node(c0, st);
                        }
                    }
                }
            }
            _ => self.unsupported.push(format!("fx-map node kind {kind}")),
        }
    }

    fn quadrant(&mut self, off: usize, w: u32, st: State) {
        let p = self.p;
        let mut o = off + 4;
        let src = (w >> 6) & 3;
        let image_flags = if w & 0x8000_0000 != 0 {
            o += 4;
            p.w(off + 4)
        } else {
            0
        };
        let has_params = (w & 0x30) != 0 || [19, 21, 23, 25, 27, 29].iter().any(|b| (w >> b) & 3 != 0) || image_flags & 2 != 0;
        let mut st = st;
        let mut vals: [Option<V4>; 7] = [None; 7];
        if has_params {
            let mut at = p.w(o) as usize;
            o += 4;
            let (k, n) = self.seed_slot(at, (w >> 4) & 3, &st);
            at = n;
            // a seed slot (absent: 0) replaces the seed, or with the inherit bit is added to the
            // parent's (lab/py/seedslot.py)
            st.seed = if (w >> 3) & 1 == 0 { k } else { st.seed.wrapping_add(k) } & M24;
            let nc = if self.color { 4 } else { 1 };
            for (k, (bit, n)) in [(19, nc), (21, 2), (23, 2), (25, 2), (27, 1), (29, 1)].into_iter().enumerate() {
                // each parameter function draws from its own LCG (SLOT_LCG)
                let (v, next) = self.slot_with(at, w >> bit, n, &st, SLOT_LCG[k]);
                vals[k] = v;
                at = next;
            }
            if image_flags & 2 != 0 {
                vals[6] = self.slot_with(at, 2, 1, &st, IMAGE_INDEX_LCG).0;
            }
        }
        let [opacity, branch, frame, size, rot, suppl, imgidx] = vals;
        let mut st2 = st;
        if let Some(b) = &branch {
            st2.branch = [st.branch[0] + b[0], st.branch[1] + b[1]];
            // the branch offset joins the quadtree centre at once (children start from centre +
            // offset, then add their quadrant offsets): the f32 summation order decides disc-edge
            // ties of far-offset stamps (concrete_smooth 23, "the 0.1545 family": 2 pixels → 0).
            // SDRT_FXBRWRAP=sum: the old separate sum of all ancestors' offsets; other test variants
            match crate::env_once!("SDRT_FXBRWRAP").as_deref().unwrap_or("merge") {
                "fract" => st2.branch = st2.branch.map(|v| v.rem_euclid(1.0)),
                "f64" => st2.branch = [(st.branch[0] as f64 + b[0] as f64) as f32, (st.branch[1] as f64 + b[1] as f64) as f32],
                "round" => st2.branch = st2.branch.map(|v| v - v.round()),
                "bwrap" => st2.branch = [st.branch[0] + (b[0] - b[0].round()), st.branch[1] + (b[1] - b[1].round())],
                m @ ("merge" | "merge_round" | "merge_fract") => {
                    st2.branch = st.branch;
                    let mut cc = [st.center[0] + b[0], st.center[1] + b[1]];
                    if m == "merge_round" { cc = cc.map(|v| v - v.round() + 0.0); }
                    if m == "merge_fract" { cc = cc.map(|v| v.rem_euclid(1.0)); }
                    st2.center = cc;
                }
                _ => {}
            }
        }
        if src != 0 {
            let s = 1.0 / (1u32 << st.depth) as f32;
            let fr = frame.unwrap_or(V4 { n: 2, a: [0.0; 4] });
            let c = match crate::env_once!("SDRT_CORDER").as_deref() {
                // test variants of the summation order (default: (centre + branch) + frame)
                Ok("1") => [st.center[0] + (st2.branch[0] + fr[0]), st.center[1] + (st2.branch[1] + fr[1])],
                Ok("2") => [(st.center[0] + fr[0]) + st2.branch[0], (st.center[1] + fr[1]) + st2.branch[1]],
                Ok("3") => [(st.center[0] as f64 + st2.branch[0] as f64 + fr[0] as f64) as f32, (st.center[1] as f64 + st2.branch[1] as f64 + fr[1] as f64) as f32],
                _ => [st2.center[0] + st2.branch[0] + fr[0], st2.center[1] + st2.branch[1] + fr[1]],
            };
            let sz = size.map_or([s, s], |v| [v[0] * s, v.get(1).copied().unwrap_or(v[0]) * s]);
            let op = opacity.unwrap_or(V4 { n: 4, a: [1.0; 4] });
            let blend_max = (w >> 18) & 1 == 1;
            let imgidx_raw = imgidx.clone();
            let index = if src == 2 { imgidx.map_or((w >> 8) & 0xF, |v| v[0] as u32) } else { (w >> 8) & 0xF };
            if crate::env_once!("SDRT_FXLOG").is_ok() {
                eprintln!("fxdraw depth {} seed {:06x} c ({:.5}, {:.5}) size ({:.5}, {:.5}) rot {:.5} op {:?}", st.depth, st.seed, c[0], c[1], sz[0], sz[1], rot.as_ref().map_or(0.0, |v| v[0]), op);
                if crate::env_once!("SDRT_FXLOG").as_deref() == Ok("x") { eprintln!("fxdrawx c {:?} sz {:?} rot {:?} roi {:?}", c, sz, rot.as_ref().map_or(0.0, |v| v[0]), self.roi); }
                if crate::env_once!("SDRT_FXLOG").as_deref() == Ok("i") { eprintln!("fxdrawi c ({:.4}, {:.4}) src {} idx {} raw {:?} op {:?}", c[0], c[1], src, index, imgidx_raw, op); }
            }
            let (rv, sv, wt) = (rot.map_or(0.0, |v| v[0]), suppl.map_or(0.0, |v| v[0]), self.weight(st.depth) * self.opmult);
            if let Some(cl) = self.collect.as_mut() {
                cl.push(Item::D((src, index, image_flags, c, sz, rv, sv, op, blend_max, wt)));
            } else if let Some(d) = self.deferred.as_mut() {
                // draws replayed at the end in their groups (see render)
                d.push((src, index, image_flags, c, sz, rv, sv, op, blend_max, wt));
            } else {
                self.draw(src, index, image_flags, c, sz, rv, sv, &op, blend_max, wt);
            }
        }
        let depth = st.depth + 1;
        if depth >= 64 {
            crate::cap_hit("fx depth");
            return;
        }
        let (m, i) = CHILD[(depth as usize - 1).min(CHILD.len() - 1)];
        let q = 1.0 / (1u32 << depth) as f32 / 2.0;
        let mut kids = [0usize; 4];
        if w & 0x10000 != 0 {
            for k in 0..4 {
                kids[k] = p.w(o + 4 * k) as usize;
            }
        } else if w & 0x20000 != 0 {
            kids = [p.w(o) as usize; 4];
        } else {
            return;
        }
        let mut cs = st.seed;
        for (k, &child) in kids.iter().enumerate() {
            cs = (m.wrapping_mul(cs).wrapping_add(i)) & M24;
            if child == 0 {
                continue;
            }
            let dx = if k % 2 == 0 { -q } else { q };
            let dy = if k < 2 { -q } else { q };
            self.node(child, State { depth, center: [st2.center[0] + dx, st2.center[1] + dy], seed: cs, branch: st2.branch, number: st.number });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn draw(&mut self, src: u32, index: u32, image_flags: u32, c: [f32; 2], sz: [f32; 2], rot: f32, suppl: f32, op: &[f32], blend_max: bool, weight: f32) {
        // SDRT_FXPROF: time spent drawing (vs evaluating the tree), printed by render()
        if !prof() {
            return self.draw_inner(src, index, image_flags, c, sz, rot, suppl, op, blend_max, weight);
        }
        let t0 = std::time::Instant::now();
        if crate::env_once!("SDRT_FXPROF").as_deref() == Ok("2") && PROF_DRAWS.load(std::sync::atomic::Ordering::Relaxed) % 100000 == 0 {
            eprintln!("fxprof draw src {} idx {} flags {} c {:?} sz {:?} roi {:?}", src, index, image_flags, c, sz, self.roi.is_some());
        }
        self.draw_inner(src, index, image_flags, c, sz, rot, suppl, op, blend_max, weight);
        PROF_DRAW_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        PROF_DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn draw_inner(&mut self, src: u32, index: u32, image_flags: u32, c: [f32; 2], sz: [f32; 2], rot: f32, suppl: f32, op: &[f32], blend_max: bool, weight: f32) {
        if !c[0].is_finite() || !c[1].is_finite() || !sz[0].is_finite() || !sz[1].is_finite() || sz[0] == 0.0 || sz[1] == 0.0 || !rot.is_finite() {
            return;
        }
        let (w, h) = (self.out.w, self.out.h);
        // early out: a stamp whose bounding extent holds no pixel centre on an axis covers nothing
        // (the coverage test needs |d| ≤ the box half-extent ≤ ext); most stamps of deep FX-Maps are
        // sub-pixel (led_stripe 8997: 1.39M draws, 24.6k pixels). Not on the ROI, mip-image or
        // SDRT_SNAP paths (unchanged order of the mip-draw counter)
        if !(src == 2 && ((image_flags >> 2) & 3) != 1) && self.no_snap {
            let ext = (sz[0].abs().max(sz[1].abs())) as f64 * 0.7072;
            if let Some((r, t)) = self.roi {
                // FX-Map space: pixel centre px at r0 + (px + ½)/w·rw; any tiling copy counts
                let (rw, rh) = ((r[1] - r[0]) as f64, (r[3] - r[2]) as f64);
                let (tw, th) = ((t[1] - t[0]) as f64, (t[3] - t[2]) as f64);
                let ext32 = (sz[0].abs().max(sz[1].abs())) * 0.7072;
                // tiling copies k0..=k1 (offset k·period), or just the stamp itself
                let copies = |cc: f32, lo: f32, period: f32, wrap: bool| -> (i64, i64, f64) {
                    if !wrap || period <= 0.0 { return (0, 0, 0.0); }
                    let k0 = ((lo - (cc + ext32)) / period).floor() as i64;
                    let k1 = ((lo + period - (cc - ext32)) / period).ceil() as i64;
                    (k0, k1, period as f64)
                };
                let any = |cc: f64, r0: f64, span: f64, n: usize| {
                    if span == 0.0 { return true; }
                    let (a, b) = ((cc - ext - r0) / span * n as f64 - 0.5, (cc + ext - r0) / span * n as f64 - 0.5);
                    let (lo, hi) = (a.min(b).ceil().max(0.0), a.max(b).floor().min(n as f64 - 1.0));
                    lo <= hi
                };
                let xs = copies(c[0], t[0], tw as f32, self.wrap.0);
                let ys = copies(c[1], t[2], th as f32, self.wrap.1);
                let hit_x = (xs.0..=xs.1).any(|k| any(c[0] as f64 + k as f64 * xs.2, r[0] as f64, rw, w));
                let hit_y = (ys.0..=ys.1).any(|k| any(c[1] as f64 + k as f64 * ys.2, r[2] as f64, rh, h));
                if !(hit_x && hit_y) {
                    return;
                }
            } else {
                let none = |cc: f32, n: usize| {
                    let (lo, hi) = ((cc as f64 - ext) * n as f64 - 0.5, (cc as f64 + ext) * n as f64 - 0.5);
                    lo.ceil() > hi.floor()
                };
                if none(c[0], w) || none(c[1], h) {
                    return;
                }
            }
        }
        let a = -rot * std::f32::consts::TAU;
        let (ca, sa) = sample_rotation(rot, src == 2 && (image_flags >> 2) & 3 == 2);
        // coverage is decided in double precision: a 45° square whose edges pass through pixel centres
        // in f32 matches sbsrender only so (scratch/rast/emu.py: f64 geometry, 0 of 25 cases differ)
        let cov = Cover::new(rot, sz);
        let ext = (sz[0].abs().max(sz[1].abs())) * 0.7072;
        // an image index past the last pattern input takes the last one (negative: the first;
        // scratch/fb/idxprobe.py: indices 2..5 of two inputs draw input 1); none connected: nothing
        if src == 2 && self.inputs.is_empty() {
            return;
        }
        let index = if src == 2 { index.min(self.inputs.len() as u32 - 1) } else { index };
        // image patterns: filtering from the image flags (bits 2-3: 0 bilinear + mips, 1 bilinear,
        // 2 nearest); the mip level is the nearest to the LOD, log2 of the longer texel derivative
        // per pixel step, its length as the reference GPU computes it (`hw_len`; rotated stamps
        // switch level up to ±0.08 LOD away from the Euclidean value; scratch/fb/mipbisect.py).
        // Measured exactly for square stamps at all angles and for 1.5:1 stamps up to 20° from an
        // axis; 1.5:1 stamps near 45° take a longer length than this (see NOTES 2026-10-30).
        // Test switches: SDRT_FXMIP=axis (the old max(tx, ty)), =cols (no row blend), =floor; SDRT_FXLEVELADD=n adds a
        // level to the n-th mipmapped draw of the run (mipbisect.py).
        let filter = (image_flags >> 2) & 3;
        let scale = self.roi.map_or(1.0, |(r, _)| (r[1] - r[0]).abs().max((r[3] - r[2]).abs()));
        let near_mip = crate::env_once!("SDRT_NEARMIP").is_ok();
        let level = if src == 2 && (filter == 0 || (filter == 2 && near_mip)) {
            let img = &self.inputs[index as usize][0];
            let tx = img.w as f32 / (sz[0].abs() / scale * w as f32);
            let ty = img.h as f32 / (sz[1].abs() / scale * h as f32);
            let mode = crate::env_once!("SDRT_FXMIP");
            let texels = if mode.as_deref() == Ok("axis") {
                tx.max(ty)
            } else {
                // texel derivatives along the screen x and y axes; near 45° non-square stamps move
                // toward the length of the larger texel scale's row (see `ROW_BLEND`)
                let (c, s) = (a.cos().abs(), a.sin().abs());
                let cols = hw_len(c * tx, s * ty).max(hw_len(s * tx, c * ty));
                let t = tx.max(ty);
                let rows = hw_len(c * t, s * t);
                if mode.as_deref() == Ok("cols") { cols } else { cols + row_blend(c.min(s).atan2(c.max(s))) * (rows - cols) }
            };
            static DRAWN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let nth = DRAWN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let bump = crate::env_once!("SDRT_FXLEVELADD").ok().and_then(|x| x.parse::<usize>().ok()) == Some(nth);
            let lod = texels.log2();
            bump as usize + if mode.as_deref() == Ok("floor") { lod.floor().max(0.0) as usize } else { lod.round().max(0.0) as usize }
        } else {
            0
        };
        let sample = Sample { src, index, filter, level, suppl, blend_max, weight };
        if let Some((r, t)) = self.roi {
            // FX-Map space → image: the render ROI (x0, x1, y0, y1) fills the image; patterns wrap
            // inside the tiling ROI only (copies a tiling period apart), nothing is drawn outside
            // it on a tiled axis (lab/py/roiprobe.py)
            let (rw, rh) = (r[1] - r[0], r[3] - r[2]);
            let (tw, th) = (t[1] - t[0], t[3] - t[2]);
            let copies = |cc: f32, lo: f32, period: f32, wrap: bool| -> Vec<f32> {
                if !wrap || period <= 0.0 {
                    return vec![0.0];
                }
                let k0 = ((lo - (cc + ext)) / period).floor() as i64;
                let k1 = ((lo + period - (cc - ext)) / period).ceil() as i64;
                (k0..=k1).map(|k| k as f32 * period).collect()
            };
            for oy in copies(c[1], t[2], th, self.wrap.1) {
                for ox in copies(c[0], t[0], tw, self.wrap.0) {
                    let (cx, cy) = (c[0] + ox, c[1] + oy);
                    let snapq = cov.quad(cx as f64, cy as f64, [r[0] as f64, r[2] as f64], [w as f64 / rw as f64, h as f64 / rh as f64]);
                    let lxq = cov.lx_quad(cx as f64, cy as f64, [r[0] as f64, r[2] as f64], [w as f64 / rw as f64, h as f64 / rh as f64]);
                    let px0 = ((((cx - ext) - r[0]) / rw * w as f32).floor() as i64 - 1).max(0);
                    let px1 = ((((cx + ext) - r[0]) / rw * w as f32).ceil() as i64 + 1).min(w as i64 - 1);
                    let py0 = ((((cy - ext) - r[2]) / rh * h as f32).floor() as i64 - 1).max(0);
                    let py1 = ((((cy + ext) - r[2]) / rh * h as f32).ceil() as i64 + 1).min(h as i64 - 1);
                    for py in py0..=py1 {
                        let fy = r[2] + (py as f32 + 0.5) / h as f32 * rh;
                        if self.wrap.1 && (fy < t[2] || fy >= t[3]) {
                            continue;
                        }
                        for px in px0..=px1 {
                            let fx = r[0] + (px as f32 + 0.5) / w as f32 * rw;
                            if self.wrap.0 && (fx < t[0] || fx >= t[1]) {
                                continue;
                            }
                            let (dx, dy) = (fx - cx, fy - cy);
                            let lx = (ca * dx - sa * dy) / (sz[0] / 2.0);
                            let ly = (sa * dx + ca * dy) / (sz[1] / 2.0);
                            let fx64 = r[0] as f64 + (px as f64 + 0.5) / w as f64 * rw as f64;
                            let fy64 = r[2] as f64 + (py as f64 + 0.5) / h as f64 * rh as f64;
                            if match &snapq { Some(q) => !q.inside(px, py), None => outside64(&cov, fx64 - cx as f64, fy64 - cy as f64) } {
                                continue;
                            }
                            let (lx, ly) = if let Some(q) = &lxq { q.local(px, py, cov.snap_lx.map_or(0, |s| s.1)) } else if cov.lx64 { cov.local(fx64 - cx as f64, fy64 - cy as f64) } else { (lx, ly) };
                            self.splat(py as usize * w + px as usize, lx, ly, &sample, op);
                        }
                    }
                }
            }
            return;
        }
        // SDRT_FXWRAPC=1: the centre wrapped into [0, 1) on tiled axes before drawing
        let c = if crate::env_once!("SDRT_FXWRAPC").is_ok() {
            [if self.wrap.0 { c[0].rem_euclid(1.0) } else { c[0] }, if self.wrap.1 { c[1].rem_euclid(1.0) } else { c[1] }]
        } else { c };
        if crate::env_once!("SDRT_FXCDBG").is_ok() && (c[0].abs() > 1.5 || c[1].abs() > 1.5) { eprintln!("fxc far centre {:?} size {:?}", c, sz); }
        let (x0, x1) = (((c[0] - ext) * w as f32 - 1.0).floor() as i64, ((c[0] + ext) * w as f32 + 1.0).ceil() as i64);
        let (y0, y1) = (((c[1] - ext) * h as f32 - 1.0).floor() as i64, ((c[1] + ext) * h as f32 + 1.0).ceil() as i64);
        let snapq = cov.quad(c[0] as f64, c[1] as f64, [0.0, 0.0], [w as f64, h as f64]);
        let lxq = cov.lx_quad(c[0] as f64, c[1] as f64, [0.0, 0.0], [w as f64, h as f64]);
        for py in y0..=y1 {
            if !self.wrap.1 && (py < 0 || py >= h as i64) {
                continue;
            }
            for px in x0..=x1 {
                if !self.wrap.0 && (px < 0 || px >= w as i64) {
                    continue;
                }
                let dx = (px as f32 + 0.5) / w as f32 - c[0];
                let dy = (py as f32 + 0.5) / h as f32 - c[1];
                let lx = (ca * dx - sa * dy) / (sz[0] / 2.0);
                let ly = (sa * dx + ca * dy) / (sz[1] / 2.0);
                let (dx64, dy64) = ((px as f64 + 0.5) / w as f64 - c[0] as f64, (py as f64 + 0.5) / h as f64 - c[1] as f64);
                if match &snapq { Some(q) => !q.inside(px, py), None => outside64(&cov, dx64, dy64) } {
                    continue;
                }
                let (lx, ly) = if let Some(q) = &lxq { q.local(px, py, cov.snap_lx.map_or(0, |s| s.1)) } else if cov.lx64 { cov.local(dx64, dy64) } else { (lx, ly) };
                let k = py.rem_euclid(h as i64) as usize * w + px.rem_euclid(w as i64) as usize;
                self.splat(k, lx, ly, &sample, op);
            }
        }
    }

    /// One pattern sample (local coordinates lx, ly in [-1, 1]) blended into output pixel k.
    fn splat(&mut self, k: usize, lx: f32, ly: f32, sm: &Sample, op: &[f32]) {
        if prof() { PROF_SPLATS.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        let Sample { src, index, filter, level, suppl, blend_max, weight } = *sm;
        // pattern sample s: generated patterns are (1, 1, 1, p) in color, p in gray;
        // images give RGBA. Measured (fxcolorfit.py).
        let s = if src == 1 {
            let p = pattern(index, lx, ly, suppl);
            [1.0, 1.0, 1.0, p]
        } else {
            let chain = &self.inputs[index as usize];
            let img = &chain[level.min(chain.len() - 1)];
            let (u, v) = ((lx + 1.0) / 2.0, (ly + 1.0) / 2.0);
            if filter == 2 { img.nearest_border(u, v) } else { img.bilinear_border(u, v, self.input_formats[index as usize]) }
        };
        if let Ok(d) = crate::env_once!("SDRT_PIXDBG") {
            if d.parse::<usize>().ok() == Some(k) {
                eprintln!("pixdbg k {} local ({:.4}, {:.4}) src {} idx {} level {} filter {} sample {:?} op {:?} w {}", k, lx, ly, src, index, level, filter, s, op, weight);
            }
        }
        let gray_from_color = src != 1 && self.inputs[index as usize][0].color;
        let dst = &mut self.out.px[k];
        if self.color {
            let a = op.get(3).copied().unwrap_or(1.0) * s[3] * weight;
            if a.is_nan() || op[..3.min(op.len())].iter().any(|x| x.is_nan()) || s.iter().any(|x| x.is_nan()) {
                return;
            }
            if self.quantized_color {
                // Quantize the premultiplied fragment first, then the blended destination. The
                // latter uses ordinary nearest rounding, not image::unorm's fixed-point rule.
                let src = [op[0] * s[0] * a, op[1] * s[1] * a, op[2] * s[2] * a, a]
                    .map(|v| crate::image::unorm(v, 8) as f32 / 255.0);
                for ch in 0..4 {
                    let d = if blend_max || ch == 3 { dst[ch] * (1.0 - src[3]) } else { dst[ch] };
                    dst[ch] = ((src[ch] + d).clamp(0.0, 1.0) * 255.0).round() / 255.0;
                }
                return;
            }
            if src == 1 && index == 15 && !blend_max {
                // gradation offset: the supplement is added after the opacity, to colour and alpha
                // (one additive layer over 0: rgb = op·op_a·(u+1)/2 + s, a = op_a·(u+1)/2 + s;
                // scratch/fb/gofsprobe.py); max blend / 8-bit colour: unmeasured
                let core = op.get(3).copied().unwrap_or(1.0) * (s[3] - suppl) * weight;
                for ch in 0..3 {
                    dst[ch] += op[ch] * core + suppl;
                }
                dst[3] += (core + suppl) * (1.0 - dst[3]);
                return;
            }
            for ch in 0..3 {
                let c = op[ch] * s[ch];
                dst[ch] = if blend_max { dst[ch] + (c - dst[ch]) * a } else { dst[ch] + c * a };
            }
            dst[3] += a * (1.0 - dst[3]);
        } else {
            // a gray FX-Map draws nothing from a color image (measured, lab/py/fxgrayimg.py)
            // Image-pattern gray values pass through a transparency complement before opacity.
            // Preserve the two f32 subtractions: replacing them with s[0] changes low bits.
            // Generated patterns/CPU function results do not take this path.
            let p = if src == 1 { s[3] } else if gray_from_color { 0.0 } else { 1.0 - (1.0 - s[0]) };
            // gradation offset: the supplement is added after the opacity, op·(u+1)/2 + s
            // (feather_generator node 64: per-strip slope = opacity, offset = supplement)
            let val = if src == 1 && index == 15 { op[0] * (p - suppl) * weight + suppl } else { op[0] * p * weight };
            if val.is_nan() {
                return;
            }
            if self.quantized_gray {
                let source = crate::image::unorm(val, 8);
                let destination = (dst[0] * 255.0).round() as u32;
                let g = if blend_max { destination.max(source) } else { (destination + source).min(255) };
                *dst = Image::gray(g as f32 / 255.0);
                return;
            }
            let g = if blend_max { dst[0].max(val) } else { dst[0] + val };
            *dst = Image::gray(g);
        }
    }
}

#[derive(Clone, Copy)]
struct Sample {
    src: u32,
    index: u32,
    filter: u32,
    level: usize,
    suppl: f32,
    blend_max: bool,
    weight: f32,
}

fn max_depth(p: &Program, off: usize, d: u32, seen: &mut Vec<usize>) -> u32 {
    if seen.len() > 100_000 {
        return d;
    }
    seen.push(off);
    let w = p.w(off);
    let kind = w & 7;
    let mut o = off + 4;
    match kind {
        0 => {
            if w & 0x8000_0000 != 0 {
                o += 4;
            }
            let ifl = if w & 0x8000_0000 != 0 { p.w(off + 4) } else { 0 };
            if (w & 0x30) != 0 || [19, 21, 23, 25, 27, 29].iter().any(|b| (w >> b) & 3 != 0) || ifl & 2 != 0 {
                o += 4;
            }
            let kids: Vec<usize> = if w & 0x10000 != 0 {
                (0..4).map(|k| p.w(o + 4 * k) as usize).collect()
            } else if w & 0x20000 != 0 {
                vec![p.w(o) as usize]
            } else {
                vec![]
            };
            kids.into_iter().filter(|&k| k != 0).map(|k| max_depth(p, k, d + 1, seen)).max().unwrap_or(d)
        }
        1 | 3 => {
            if (w >> 4) & 3 != 0 {
                o += 4;
            }
            if w & 0x80 != 0 {
                o += 4;
            }
            let nk = if kind == 1 || w & 0x40 != 0 { 2 } else { 1 };
            (0..nk).map(|k| p.w(o + 4 * k) as usize).filter(|&k| k != 0).map(|k| max_depth(p, k, d, seen)).max().unwrap_or(d)
        }
        _ => d,
    }
}

/// What the tree can draw: generated pattern kinds, whether it draws images, whether any image
/// quadrant takes its index from a function (anywhere, taken branch or not).
fn scan_groups(p: &Program, off: usize, seen: &mut std::collections::HashSet<usize>, acc: &mut (std::collections::BTreeSet<u32>, bool, bool)) {
    if off == 0 || !seen.insert(off) || seen.len() > 100_000 {
        return;
    }
    let w = p.w(off);
    let kind = w & 7;
    let mut o = off + 4;
    match kind {
        0 => {
            let ifl = if w & 0x8000_0000 != 0 { o += 4; p.w(off + 4) } else { 0 };
            match (w >> 6) & 3 {
                1 => { acc.0.insert((w >> 8) & 0xF); }
                2 => { acc.1 = true; if ifl & 2 != 0 { acc.2 = true; } }
                _ => {}
            }
            if (w & 0x30) != 0 || [19, 21, 23, 25, 27, 29].iter().any(|b| (w >> b) & 3 != 0) || ifl & 2 != 0 {
                o += 4;
            }
            let kids: Vec<usize> = if w & 0x10000 != 0 {
                (0..4).map(|k| p.w(o + 4 * k) as usize).collect()
            } else if w & 0x20000 != 0 {
                vec![p.w(o) as usize]
            } else {
                vec![]
            };
            for k in kids { scan_groups(p, k, seen, acc); }
        }
        1 | 3 => {
            if (w >> 4) & 3 != 0 { o += 4; }
            if w & 0x80 != 0 { o += 4; }
            let nk = if kind == 1 || w & 0x40 != 0 { 2 } else { 1 };
            for k in 0..nk { scan_groups(p, p.w(o + 4 * k) as usize, seen, acc); }
        }
        _ => {}
    }
}

/// SDRT_FXPROF set (read once): timers and counters on
fn prof() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| crate::env_once!("SDRT_FXPROF").is_ok())
}
static PROF_DRAW_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_SPLATS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// SDRT_FXPROF=2: evaluations per function offset: (count, instructions, pure)
static PROF_FN: std::sync::Mutex<std::collections::BTreeMap<usize, (u64, usize, bool)>> = std::sync::Mutex::new(std::collections::BTreeMap::new());
pub fn prof_fn_report() {
    let m = PROF_FN.lock().unwrap();
    let mut v: Vec<_> = m.iter().map(|(o, e)| (e.0, *o, e.1, e.2)).collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    let total: u64 = v.iter().map(|e| e.0).sum();
    let pure: u64 = v.iter().filter(|e| e.3).map(|e| e.0).sum();
    eprintln!("fxprof functions: {} evals, {} of them pure ({:.1}%)", total, pure, 100.0 * pure as f64 / total.max(1) as f64);
    for (n, o, k, p) in v.iter().take(25) {
        eprintln!("  fn {:>10} evals {:>11} ins {:>3} pure {}", o, n, k, p);
    }
}
static PROF_EVAL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_EVALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_INS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Evaluate the tree in parallel: the top is walked recording draws and subtrees (tasks, with the
/// variables at that point) in traversal order; the tasks run on rayon workers, each recording its
/// draws; the records are spliced back in order and drawn as a sequential traversal would (or
/// grouped, as usual). Each quadrant's random state comes from its parent's seed, so a subtree is
/// a function of its state and the variables it reads. Every task starts from the variables the
/// top walk had at its place; that is what a sequential run gives it unless it reads (before
/// writing it itself) a variable an earlier task wrote, or the top walk after the first task
/// touches one a task wrote — checked on the recorded accesses (a copy that sets and then gets its
/// own variables is fine). Otherwise the walk goes one level deeper (the setting moves into the
/// top walk), up to a limit, and then the caller traverses sequentially.
/// Returns false when nothing was done. SDRT_FXPAR=0: always sequential; SDRT_FXPARDBG=1 logs.
fn traverse_parallel(ctx: &mut Ctx, root_off: usize, root: State, node: usize) -> bool {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    if crate::env_once!("SDRT_FXPAR").as_deref() == Ok("0") || crate::env_once!("SDRT_FXLOG").is_ok() || crate::env_once!("SDRT_ITERREV").is_ok() {
        return false;
    }
    let threads = rayon::current_num_threads();
    if threads < 2 {
        return false;
    }
    let dbg = crate::env_once!("SDRT_FXPARDBG").is_ok();
    let vars0 = ctx.vars.borrow().clone();
    let unsup0 = ctx.unsupported.len();
    let mut last_tasks = 0;
    let union = |a: &mut Vec<u16>, b: &[u16]| {
        for x in b {
            if !a.contains(x) {
                a.push(*x);
            }
        }
    };
    for split in 1..=8u32 {
        *ctx.vars.borrow_mut() = vars0.clone();
        ctx.unsupported.truncate(unsup0);
        ctx.collect = Some(vec![]);
        ctx.snaps.clear();
        ctx.snap_budget = 8 << 20;
        ctx.track = None;
        ctx.split_at = split;
        ctx.call_depth = 0;
        ctx.node(root_off, root);
        let mut items = ctx.collect.take().unwrap();
        ctx.split_at = u32::MAX;
        if ctx.snap_budget == 0 {
            // too many distinct variable states to keep: sequential
            if dbg {
                eprintln!("fxpar node {} split {}: variable snapshots over budget, sequential", node, split);
            }
            ctx.snaps.clear();
            break;
        }
        let top = ctx.track.take().map(|t| t.into_inner()).unwrap_or_default();
        let snaps = std::mem::take(&mut ctx.snaps);
        let tasks: Vec<(usize, usize, State, u32)> = items.iter().enumerate().filter_map(|(i, it)| if let Item::T(o, s, v) = it { Some((i, *o, *s, *v)) } else { None }).collect();
        if dbg {
            eprintln!("fxpar node {} split {} items {} tasks {} variable states {}", node, split, items.len(), tasks.len(), snaps.len());
        }
        // too few tasks to share: one level deeper, while that still adds tasks
        if !tasks.is_empty() && tasks.len() < 4 * threads && split < 8 && (split < 3 || tasks.len() > last_tasks) {
            last_tasks = tasks.len();
            continue;
        }
        // the tasks in windows (memory: a window's draws, not the whole tree's — castle_wall 805
        // has 17M stamps): each window runs in parallel, is checked, then spliced and drawn in
        // traversal order. A failed check rolls everything back (the target image or the grouped
        // list) and the walk goes one level deeper.
        let backup = if ctx.deferred.is_none() { Some(ctx.out.clone()) } else { None };
        let def0 = ctx.deferred.as_ref().map(|d| d.len());
        let window = (threads * 64).max(1);
        let nworkers = (threads * 4).min(tasks.len()).max(1);
        let runaway = std::sync::Arc::new(AtomicBool::new(false));
        let mut workers: Vec<Ctx> = (0..nworkers).map(|_| { let mut w = ctx.worker(); w.spec = Some(runaway.clone()); w }).collect();
        let mut written: Vec<u16> = vec![];
        let mut ok = true;
        let mut next_item = 0usize;
        let mut items: Vec<Option<Item>> = items.drain(..).map(Some).collect();
        let splice = |ctx: &mut Ctx, recs: Vec<DrawRec>| {
            for (src, index, fl, c, sz, rv, sv, op, bm, wt) in recs {
                if let Some(d) = ctx.deferred.as_mut() {
                    if ctx.gpu_collect && d.len() >= GPU_DRAW_LIMIT {
                        return;
                    }
                    d.push((src, index, fl, c, sz, rv, sv, op, bm, wt));
                } else {
                    ctx.draw(src, index, fl, c, sz, rv, sv, &op, bm, wt);
                }
            }
        };
        for win in tasks.chunks(window) {
            if ctx.gpu_collect && ctx.deferred.as_ref().is_some_and(|d| d.len() >= GPU_DRAW_LIMIT) {
                break;
            }
            let per = win.len().div_ceil(nworkers).max(1);
            let chunks: Vec<&[(usize, usize, State, u32)]> = win.chunks(per).collect();
            let abort = AtomicBool::new(false);
            for w in workers.iter_mut() {
                w.work = 0;
            }
            // per chunk: the tasks' draws, the variables read from outside the chunk, written
            type ChunkOut = (Vec<(usize, Vec<DrawRec>)>, Vec<u16>, Vec<u16>, Vec<String>);
            let results: Vec<ChunkOut> = workers[..chunks.len()]
                .par_iter_mut()
                .zip(chunks.into_par_iter())
                .map(|(w, chunk)| {
                    let (mut out, mut ext, mut wrote) = (Vec::with_capacity(chunk.len()), vec![], vec![]);
                    for &(i, off, st, si) in chunk {
                        if abort.load(Ordering::Relaxed) {
                            break;
                        }
                        w.vars.borrow_mut().clone_from(&snaps[si as usize]);
                        *w.track.as_ref().unwrap().borrow_mut() = VarTrack::default();
                        w.node(off, st);
                        let t = w.track.as_ref().unwrap().borrow();
                        if t.read_ext.iter().any(|v| wrote.contains(v)) {
                            abort.store(true, Ordering::Relaxed);
                            break;
                        }
                        union(&mut ext, &t.read_ext);
                        union(&mut wrote, &t.written);
                        drop(t);
                        out.push((i, std::mem::take(w.deferred.as_mut().unwrap())));
                    }
                    (out, ext, wrote, std::mem::take(&mut w.unsupported))
                })
                .collect();
            if runaway.load(Ordering::Relaxed) {
                ok = false;
                break;
            }
            ok = !abort.load(Ordering::Relaxed);
            if ok {
                for (_, ext, wrote, _) in &results {
                    if ext.iter().any(|v| written.contains(v)) {
                        ok = false;
                        break;
                    }
                    union(&mut written, wrote);
                }
            }
            if ok && top.read_ext.iter().chain(&top.written).any(|v| written.contains(v)) {
                ok = false;
            }
            if !ok {
                break;
            }
            // splice: the items up to this window's last task, in order
            let last = win.last().unwrap().0;
            let mut sub: Vec<(usize, Vec<DrawRec>)> = vec![];
            for (out, _, _, uns) in results {
                sub.extend(out);
                ctx.unsupported.extend(uns);
            }
            sub.sort_by_key(|e| e.0);
            let mut sub = sub.into_iter().peekable();
            for i in next_item..=last {
                match items[i].take() {
                    Some(Item::D(r)) => splice(ctx, vec![r]),
                    Some(Item::T(..)) => {
                        let recs = if sub.peek().is_some_and(|e| e.0 == i) { sub.next().unwrap().1 } else { vec![] };
                        splice(ctx, recs);
                    }
                    None => {}
                }
            }
            next_item = last + 1;
        }
        if dbg {
            eprintln!("fxpar node {} {} tasks in windows of {}: variables written by tasks {:?}, independent {}", node, tasks.len(), window, written, ok);
        }
        if !ok {
            // roll back what earlier windows drew or grouped
            if let Some(b) = backup {
                ctx.out = b;
            }
            if let (Some(d), Some(n0)) = (ctx.deferred.as_mut(), def0) {
                d.truncate(n0);
            }
            if runaway.load(Ordering::Relaxed) {
                // a speculative task ran away: sequential, no deeper attempt
                if dbg {
                    eprintln!("fxpar node {}: a speculative task exceeded its budget, sequential", node);
                }
                break;
            }
            last_tasks = tasks.len();
            continue;
        }
        // the items after the last task
        for i in next_item..items.len() {
            if let Some(Item::D(r)) = items[i].take() {
                splice(ctx, vec![r]);
            }
        }
        return true;
    }
    *ctx.vars.borrow_mut() = vars0;
    ctx.unsupported.truncate(unsup0);
    false
}

pub fn render(rt: &mut Runtime, n: &Node, seed: u32) -> Image {
    match render_inner(rt, n, seed, false) {
        Rendered::Cpu(im) => im,
        Rendered::Gpu(_) => unreachable!(),
    }
}

pub struct GpuPlan {
    pub background: Image,
    pub tables: Vec<Image>,
    pub tiles_x: u32,
    pub roi: Option<[f32; 4]>,
}

enum Rendered {
    Cpu(Image),
    Gpu(Result<GpuPlan, String>),
}

#[cfg(feature = "gpu")]
pub fn prepare_gpu(rt: &mut Runtime, n: &Node, seed: u32) -> Result<GpuPlan, String> {
    let t0 = std::time::Instant::now();
    let res = match render_inner(rt, n, seed, true) {
        Rendered::Gpu(p) => p,
        Rendered::Cpu(_) => unreachable!(),
    };
    if prof() || crate::env_once!("SDRT_TIME").is_ok() {
        eprintln!("fxnode {} prepare_gpu total: {:.3}s", n.index, t0.elapsed().as_secs_f64());
    }
    res
}

fn render_inner(rt: &mut Runtime, n: &Node, seed: u32, gpu_collect: bool) -> Rendered {
    let t_render = std::time::Instant::now();
    let (d0, n0, s0) = (PROF_DRAW_NS.load(std::sync::atomic::Ordering::Relaxed), PROF_DRAWS.load(std::sync::atomic::Ordering::Relaxed), PROF_SPLATS.load(std::sync::atomic::Ordering::Relaxed));
    if crate::env_once!("SDRT_FXLOG").is_ok() {
        eprintln!("fxnode {}", n.index);
    }
    let (w, h) = n.size();
    let bg = rt.param(n, "background", seed);
    let hurst = rt.param(n, "hurst", seed).map_or(1.0, |v| v[0]);
    let opmult = rt.param(n, "opacitymult", seed).map_or(1.0, |v| v[0]);
    let four = |v: Vec<f32>| [v[0], v.get(1).copied().unwrap_or(v[0]), v.get(2).copied().unwrap_or(v[0]), v.get(3).copied().unwrap_or(v[0])];
    let roi = rt.param(n, "renderroi", seed).map(|r| {
        let t = rt.param(n, "tilingroi", seed).map_or([0.0, 1.0, 0.0, 1.0], four);
        (four(r), t)
    });
    let fill = if n.color { bg.as_ref().map_or([0.0; 4], |v| [v[0], v[1], v[2], v[3]]) } else { Image::gray(bg.as_ref().map_or(0.0, |v| v[0])) };
    let mut out = Image::new(w, h, n.color, fill);
    if let crate::program::Slot::Input(k) = n.param("background") {
        out = rt.images[*k].as_ref().map_or(out.clone(), |im| im.resized(w, h));
    }
    let quantized_color = n.color && n.format() == 0 && crate::env_once!("SDRT_FX8_QUANT").as_deref() != Ok("0");
    let quantized_gray = !n.color && n.format() == 0 && crate::env_once!("SDRT_FX8_QUANT").as_deref() != Ok("0");
    if quantized_color || quantized_gray { out.quantize(0); }
    let inputs: Vec<Vec<Image>> = n
        .inputs
        .iter()
        .map(|&k| {
            let img = rt.images[k].as_deref().cloned().unwrap_or(Image::new(1, 1, false, [0.0; 4]));
            crate::image::mips(&img, rt.p.nodes[k].format())
        })
        .collect();
    let md = max_depth(rt.p, n.data, 0, &mut vec![]);
    let rtr: &Runtime = rt;
    let input_formats = n.inputs.iter().map(|&k| rtr.p.nodes[k].format()).collect();
    // Draw order (scratch/fb/fxorder*.py, alpha blending; roadsubstance001 2603 exact): stamps are
    // drawn in groups — generated patterns by kind ascending (a disc drawn first still ends above
    // a square), then images; images keep traversal order unless any image quadrant of the tree
    // (even on an untaken switch branch) takes its index from a function, then by index ascending.
    // Traversal order within a group. Draws are deferred only when there is more than one group.
    // SDRT_FXORDER=trav: plain traversal order (old)
    let mut acc = (std::collections::BTreeSet::new(), false, false);
    scan_groups(rt.p, n.data, &mut std::collections::HashSet::new(), &mut acc);
    let (kinds, has_img, dyn_img) = acc;
    if gpu_collect {
        let reason = if out.color != n.color { Some("background colour differs from node") }
            else if kinds.contains(&4) { Some("Gaussian requires CPU expf") }
            else if out.px.iter().flatten().any(|x| !x.is_finite()) { Some("non-finite background") }
            else if let Some((r, t)) = roi {
                if r.iter().chain(t.iter()).any(|x| !x.is_finite() || x.abs() > 16.0)
                    || r[1] - r[0] < 1.0 / 65536.0 || r[3] - r[2] < 1.0 / 65536.0 {
                    Some("non-finite, reversed, degenerate or excessive ROI")
                } else {
                    None
                }
            } else { None };
        if let Some(why) = reason { return Rendered::Gpu(Err(why.into())); }
    }
    let ngroups = kinds.len() + has_img as usize;
    let regroup = crate::env_once!("SDRT_FXORDER").as_deref() != Ok("trav") && (ngroups > 1 || dyn_img);
    let mut ctx = Ctx { gpu_collect, deferred: if regroup || gpu_collect { Some(vec![]) } else { None }, collect: None, snaps: vec![], snap_budget: 0, spec: None, work: 0, visits: 0, track: None, call_depth: 0, split_at: u32::MAX, no_snap: crate::env_once!("SDRT_SNAP").is_err() && crate::env_once!("SDRT_SNAPLX").is_err() && crate::env_once!("SDRT_FXWRAPC").is_err(), rt: rtr, p: rtr.p, out, color: n.color, quantized_color, quantized_gray, max_depth: md, hurst, opmult, roi, wrap: (n.b3 & 1 != 0, n.b3 & 2 != 0), inputs: std::sync::Arc::new(inputs), input_formats, funcs: Default::default(), weights: vec![], bufs: Default::default(), vars: std::cell::RefCell::new(rtr.node_vars.clone()), unsupported: vec![] };
    ctx.weights = (0..64).map(|d| ctx.weight_calc(d)).collect();
    let root = State { depth: 0, center: [0.5, 0.5], seed: seed & M24, branch: [0.0, 0.0], number: 0.0 };
    if gpu_collect && ctx.estimate_draws(n.data, root, GPU_DRAW_LIMIT) >= GPU_DRAW_LIMIT {
        return Rendered::Gpu(Err("draw collection budget".into()));
    }
    let par = traverse_parallel(&mut ctx, n.data, root, n.index);
    if crate::env_once!("SDRT_TIME").is_ok() {
        eprintln!("fxnode {} traverse_parallel: {}", n.index, par);
    }
    if !par {
        ctx.node(n.data, root);
    }
    if let Some(mut d) = ctx.deferred.take() {
        let nin = ctx.inputs.len().max(1) as u32;
        if regroup { d.sort_by_key(|e| if e.0 == 2 { (1u32, if dyn_img { e.1.min(nin - 1) } else { 0 }) } else { (0, e.1) }); }
        if gpu_collect {
            let t_plan = std::time::Instant::now();
            let plan = gpu_plan(&ctx, &d);
            if crate::env_once!("SDRT_TIME").is_ok() {
                eprintln!("fxnode {} gpu_plan: {:.3}s, draws: {}, result: {:?}", n.index, t_plan.elapsed().as_secs_f64(), d.len(), plan.as_ref().err());
            }
            if plan.is_ok() {
                for u in ctx.unsupported { rt.unsupported.push((n.index, u)); }
            }
            return Rendered::Gpu(plan);
        }
        for (src, index, fl, c, sz, rv, sv, op, bm, wt) in d {
            ctx.draw(src, index, fl, c, sz, rv, sv, &op, bm, wt);
        }
    }
    let Ctx { mut out, unsupported, .. } = ctx;
    if !n.color && n.format() == 2 {
        // Gray half targets accumulate in f32, then truncate to half. Prequantizing the
        // background or rounding the final half to nearest both fail the random-layer probe.
        for p in &mut out.px { *p = Image::gray(crate::image::half_trunc(p[0])); }
    } else if n.color && n.format() == 2 && crate::env_once!("SDRT_FXHALFC").as_deref() != Ok("rne") {
        // color half targets too (feather_generator 48, max-blended points, frozen inputs:
        // one half ulp on a quarter of the pixels with nearest-even)
        for p in &mut out.px { *p = p.map(crate::image::half_trunc); }
    }
    for u in unsupported {
        rt.unsupported.push((n.index, u));
    }
    if prof() {
        let o = std::sync::atomic::Ordering::Relaxed;
        eprintln!("fxprof node {} {}x{}: total {:.3}s draw {:.3}s draws {} splats {} | evals {} ({:.3}s, {} instructions, cumulative)", n.index, w, h, t_render.elapsed().as_secs_f64(),
            (PROF_DRAW_NS.load(o) - d0) as f64 * 1e-9, PROF_DRAWS.load(o) - n0, PROF_SPLATS.load(o) - s0,
            PROF_EVALS.load(o), PROF_EVAL_NS.load(o) as f64 * 1e-9, PROF_INS.load(o));
    }
    Rendered::Cpu(out)
}

/// Partition only the already ordered draw list. A failed planning attempt has no
/// output/VM side effects; retrying halves therefore preserves every per-pixel operation.
fn gpu_plan(ctx: &Ctx, draws: &[DrawRec]) -> Result<GpuPlan, String> {
    if draws.len() >= GPU_DRAW_LIMIT { return Err("draw collection budget".into()); }
    let all_len = draws.len();
    // In Substance FX-Map, stamps occupy at least 8-12 vec4s each in the tile tables.
    // Trees with >1.25M draws cannot fit in the 16,000,000 streamed table limit.
    if all_len > 1_250_000 {
        return Err("streamed tile table budget".into());
    }
    fn collect(ctx: &Ctx, draws: &[DrawRec], all_len: usize, tables: &mut Vec<Image>, total: &mut usize, done: &mut usize) -> Result<(), String> {
        if draws.len() > 400_000 {
            let mid = draws.len() / 2;
            collect(ctx, &draws[..mid], all_len, tables, total, done)?;
            return collect(ctx, &draws[mid..], all_len, tables, total, done);
        }
        match gpu_plan_chunk(ctx, draws) {
            Ok(table) => {
                *done += draws.len();
                *total += table.px.len();
                if *total > 16_000_000 || tables.len() >= 64 { return Err("streamed tile table budget".into()); }
                tables.push(table);
                if *done > 0 {
                    let ratio = all_len as f64 / *done as f64;
                    if (tables.len() as f64 * ratio) > 64.0 || (*total as f64 * ratio) > 15_200_000.0 {
                        return Err("streamed tile table budget".into());
                    }
                }
                Ok(())
            }
            Err(e) if draws.len() > 1 && (e == "tile table budget" || e == "image packing budget") => {
                let mid = draws.len() / 2;
                collect(ctx, &draws[..mid], all_len, tables, total, done)?;
                collect(ctx, &draws[mid..], all_len, tables, total, done)
            }
            Err(e) => Err(e),
        }
    }
    let mut tables = vec![];
    collect(ctx, draws, all_len, &mut tables, &mut 0, &mut 0)?;
    Ok(GpuPlan {background:ctx.out.clone(),tables,tiles_x:ctx.out.w.div_ceil(16) as u32,
        roi:ctx.roi.map(|(r,_)|[r[0],r[2],r[1]-r[0],r[3]-r[2]])})
}

/// Exact CPU f64 coverage as rectangles plus rotated row intervals. Pixel sampling
/// and ordered blending run on the GPU. Each packed table is below a storage binding.
fn gpu_plan_chunk(ctx: &Ctx, draws: &[DrawRec]) -> Result<Image, String> {
    if ctx.out.px.iter().flatten().any(|x| !x.is_finite()) { return Err("non-finite background".into()); }
    if let Some((r,t))=ctx.roi {
        if r.iter().chain(t.iter()).any(|x| !x.is_finite() || x.abs()>16.0)
            || r[1]-r[0]<1.0/65536.0 || r[3]-r[2]<1.0/65536.0 {
            return Err("non-finite, reversed, degenerate or excessive ROI".into());
        }
    }
    let (w,h)=(ctx.out.w,ctx.out.h);
    let (tx,ty)=(w.div_ceil(16),h.div_ceil(16));
    let mut bins=vec![Vec::<u32>::new();tx*ty];
    let mut table=vec![[0.0;4];tx*ty];
    let mut references=0usize;
    let words=|v:[u32;4]| v.map(f32::from_bits);
    let mut packed=std::collections::HashMap::new();
    for &(src,kind,image_flags,c,sz,rot,suppl,op,bm,weight) in draws {
        if !c.iter().chain(sz.iter()).all(|x|x.is_finite()) || sz.contains(&0.0) || !rot.is_finite() { continue; }
        if !matches!(src,1|2) || (src==1 && kind==4) { return Err("unsupported pattern".into()); }
        if src==2 && ctx.inputs.is_empty() {continue;}
        if c.iter().any(|x|x.abs()>16384.0) || sz.iter().any(|x|x.abs()>64.0 || x.abs()<2.0f32.powi(-30)) {
            return Err("stamp size or centre exceeds finite geometry budget".into());
        }
        if op.iter().chain([&suppl,&weight]).any(|x| !x.is_finite() || x.abs()>65536.0) {
            return Err("non-finite or excessive opacity/supplement/weight".into());
        }
        let mut image_desc=[0u32;4];
        if src==2 {
            let index=(kind as usize).min(ctx.inputs.len()-1);
            let chain=&ctx.inputs[index];
            let filter=(image_flags>>2)&3;
            // Same f32 LOD calculation as draw_inner; mip construction and VM samplers
            // are shared with the CPU path. Experimental LOD modes are rejected by ops.
            let level=if filter==0 {
                let img=&chain[0];
                let scale=ctx.roi.map_or(1.0,|(r,_)|(r[1]-r[0]).abs().max((r[3]-r[2]).abs()));
                let tx=img.w as f32/(sz[0].abs()/scale*w as f32);
                let ty=img.h as f32/(sz[1].abs()/scale*h as f32);
                let a=-rot*std::f32::consts::TAU;
                let (c,s)=(a.cos().abs(),a.sin().abs());
                let cols=hw_len(c*tx,s*ty).max(hw_len(s*tx,c*ty));
                let t=tx.max(ty);
                let rows=hw_len(c*t,s*t);
                let texels=cols+row_blend(c.min(s).atan2(c.max(s)))*(rows-cols);
                (texels.log2().round().max(0.0) as usize).min(chain.len()-1)
            } else {0};
            let img=&chain[level];
            let offset=if let Some(&offset)=packed.get(&(index,level)) {offset} else {
                if table.len()+references+img.px.len()>4_000_000 {return Err("image packing budget".into());}
                if img.px.iter().flatten().any(|x|!x.is_finite() || x.abs()>65536.0) {
                    return Err("non-finite or excessive image texel".into());
                }
                let offset=table.len() as u32;
                table.extend(img.px.iter().map(|p|if img.color {*p} else {Image::gray(p[0])}));
                packed.insert((index,level),offset);
                offset
            };
            image_desc=[offset,img.w as u32,img.h as u32,ctx.input_formats[index]|((img.color as u32)<<4)|(filter<<8)];
        }
        let cov=Cover::new(rot,sz);
        let (sample_ca,sample_sa)=sample_rotation(rot,src==2 && (image_flags>>2)&3==2);
        let ext=sz[0].abs().max(sz[1].abs())*0.7072;
        let mut append=|bounds:[usize;4],center:[f32;2],ox:i64,oy:i64| -> Result<(),String> {
            let [l,r,t,b]=bounds;
            let offset=table.len() as u32;
            table.extend([words([l as u32,r as u32,t as u32,b as u32]),
                [center[0],center[1],sz[0]/2.0,sz[1]/2.0],words([ox as u32,oy as u32,if src==2 {16} else {kind},u32::from(bm)]),
                [op[0],op.get(1).copied().unwrap_or(0.0),op.get(2).copied().unwrap_or(0.0),op.get(3).copied().unwrap_or(1.0)],
                [weight,suppl,sample_sa,sample_ca],
                words([0,0,0,0]),words(image_desc)]);
            if rot != 0.0 {
                // WGSL has no f64. Keep the exact CPU coverage predicates on the host,
                // intersecting their monotonic x intervals once per row. Sampling and
                // all ordered blending stay on the GPU.
                table[offset as usize+5]=words([table.len() as u32,1,0,0]);
                for y in t..=b {
                    let point=|x:usize| {
                        if let Some((r,_))=ctx.roi {
                            (r[0] as f64+(x as f64+0.5)/w as f64*(r[1]-r[0]) as f64-center[0] as f64,
                             r[2] as f64+(y as f64+0.5)/h as f64*(r[3]-r[2]) as f64-center[1] as f64)
                        } else {
                            (((x as i64+ox) as f64+0.5)/w as f64-center[0] as f64,
                             ((y as i64+oy) as f64+0.5)/h as f64-center[1] as f64)
                        }
                    };
                    let lower=|pred:&dyn Fn(usize)->bool| {
                        let (mut lo,mut hi)=(l,r+1);
                        while lo<hi {let m=(lo+hi)/2;if pred(m) {hi=m;} else {lo=m+1;}}
                        lo
                    };
                    let (mut lo,mut end)=(l,r+1);
                    for axis in 0..2 {
                        let slope=if axis==0 {cov.ca/cov.hx} else {cov.sa/cov.hy};
                        let value=|x| {let (dx,dy)=point(x);
                            let v=if axis==0 {(cov.ca*dx-cov.sa*dy)/cov.hx} else {(cov.sa*dx+cov.ca*dy)/cov.hy};
                            if slope<0.0 {-v} else {v}
                        };
                        lo=lo.max(lower(&|x|value(x)>= -1.0-1e-6));
                        end=end.min(lower(&|x|value(x)>1.0+1e-6));
                    }
                    table.push(words([lo as u32,end as u32,0,0]));
                }
            }
            for y in t/16..=b/16 {for x in l/16..=r/16 {bins[y*tx+x].push(offset); references+=1;}}
            if table.len()+references>4_000_000 {return Err("tile table budget".into());}
            Ok(())
        };
        if let Some((r,t))=ctx.roi {
            let (rw,rh)=(r[1]-r[0],r[3]-r[2]);
            let (tw,th)=(t[1]-t[0],t[3]-t[2]);
            let copies=|cc:f32,lo:f32,period:f32,wrap:bool| -> Result<(i64,i64,f32),String> {
                if !wrap || period<=0.0 {return Ok((0,0,0.0));}
                let k0=((lo-(cc+ext))/period).floor() as i64;
                let k1=((lo+period-(cc-ext))/period).ceil() as i64;
                if k1.saturating_sub(k0)>4096 {return Err("ROI copy budget".into());}
                Ok((k0,k1,period))
            };
            let xs=copies(c[0],t[0],tw,ctx.wrap.0)?;
            let ys=copies(c[1],t[2],th,ctx.wrap.1)?;
            if (xs.1-xs.0+1)*(ys.1-ys.0+1)>4096 {return Err("ROI copy budget".into());}
            // Keep draw_inner's f64 early rejection, before rounded f32 copy centres.
            let early=|cc:f32,r0:f32,span:f32,n:usize,copies:(i64,i64,f32)| {
                let ext=sz[0].abs().max(sz[1].abs()) as f64*0.7072;
                (copies.0..=copies.1).any(|k| {
                    let cc=cc as f64+k as f64*copies.2 as f64;
                    let a=(cc-ext-r0 as f64)/span as f64*n as f64-0.5;
                    let b=(cc+ext-r0 as f64)/span as f64*n as f64-0.5;
                    a.min(b).ceil().max(0.0)<=a.max(b).floor().min(n as f64-1.0)
                })
            };
            if !(src==2 && ((image_flags>>2)&3)!=1)
                && (!early(c[0],r[0],rw,w,xs) || !early(c[1],r[2],rh,h,ys)) {continue;}
            let axis=|cc:f32,r0:f32,span:f32,t0:f32,t1:f32,n:usize,half:f64,wrap:bool| -> Option<(usize,usize)> {
                // Positive spans make coverage and tiling predicates monotonic. Binary searches
                // evaluate exactly the original f64 coverage / f32 tiling expressions.
                let lower=|pred:&dyn Fn(usize)->bool| {
                    let (mut lo,mut hi)=(0,n);
                    while lo<hi {let m=(lo+hi)/2;if pred(m) {hi=m;} else {lo=m+1;}}
                    lo
                };
                let coverage=|p:usize| (r0 as f64+(p as f64+0.5)/n as f64*span as f64-cc as f64)/half;
                let mut lo=lower(&|p|coverage(p)> -1.0+1e-6);
                let mut end=lower(&|p|coverage(p)>1.0+1e-6);
                let px0=((((cc-ext)-r0)/span*n as f32).floor() as i64-1).max(0);
                let px1=((((cc+ext)-r0)/span*n as f32).ceil() as i64+1).min(n as i64-1);
                lo=lo.max(px0 as usize); end=end.min((px1+1).max(0) as usize);
                if wrap {
                    let point=|p:usize|r0+(p as f32+0.5)/n as f32*span;
                    lo=lo.max(lower(&|p|point(p)>=t0)); end=end.min(lower(&|p|point(p)>=t1));
                }
                (lo<end).then(||(lo,end-1))
            };
            for y in ys.0..=ys.1 {for x in xs.0..=xs.1 {
                let center=[c[0]+x as f32*xs.2,c[1]+y as f32*ys.2];
                let Some((l,rr))=axis(center[0],r[0],rw,t[0],t[1],w,cov.ex,ctx.wrap.0) else {continue};
                let Some((top,b))=axis(center[1],r[2],rh,t[2],t[3],h,cov.ey,ctx.wrap.1) else {continue};
                append([l,rr,top,b],center,0,0)?;
            }}
        } else {
            let axis=|cc:f32, size:usize, half:f64, wrap:bool| -> Option<(i64,i64)> {
                // Match draw_inner's conservative f32 bounding box before applying coverage.
                let mut lo=(((cc-ext)*size as f32-1.0).floor() as i64).max(((cc as f64-half)*size as f64-0.5).floor() as i64-2);
                let mut hi=(((cc+ext)*size as f32+1.0).ceil() as i64).min(((cc as f64+half)*size as f64-0.5).ceil() as i64+2);
                let inside=|p:i64| {let b=((p as f64+0.5)/size as f64-cc as f64)/half;b> -1.0+1e-6 && b<=1.0+1e-6};
                while lo<=hi && !inside(lo) {lo+=1;}
                while hi>=lo && !inside(hi) {hi-=1;}
                if !wrap {lo=lo.max(0);hi=hi.min(size as i64-1);}
                (lo<=hi).then_some((lo,hi))
            };
            let Some((x0,x1))=axis(c[0],w,cov.ex,ctx.wrap.0) else {continue};
            let Some((y0,y1))=axis(c[1],h,cov.ey,ctx.wrap.1) else {continue};
            for cy in y0.div_euclid(h as i64)..=y1.div_euclid(h as i64) {
                for cx in x0.div_euclid(w as i64)..=x1.div_euclid(w as i64) {
                    let (ox,oy)=(cx*w as i64,cy*h as i64);
                    append([(x0-ox).max(0) as usize,(x1-ox).min(w as i64-1) as usize,
                        (y0-oy).max(0) as usize,(y1-oy).min(h as i64-1) as usize],c,ox,oy)?;
                }
            }
        }
    }
    for (k,bin) in bins.into_iter().enumerate() {
        table[k]=words([table.len() as u32,bin.len() as u32,0,0]);
        table.extend(bin.into_iter().map(|i|words([i,0,0,0])));
    }
    Ok(Image {w:table.len(),h:1,color:true,px:table})
}
