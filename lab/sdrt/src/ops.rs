//! Executing a program node by node. Each op follows research/sbsar/NOTES.md; formulas that are
//! not measured yet are marked TODO and listed by `Runtime::unsupported`.

use std::collections::HashMap;

use crate::image::Image;
use crate::program::{Node, Program, Slot};
use crate::vm::{Env, Eval, Func, Value};

const NODE_A: u32 = 16294893;
const NODE_C: u32 = 10493233;
pub const QUAD_A: u32 = 3626811;
pub const QUAD_C: u32 = 51137;
const M24: u32 = 0xFF_FFFF;

/// Filter kernels (`Gpu::dispatch_passes` prepends the prelude): the shared sampling helpers of
/// gpu_sample.wgsl, then the kernel.
#[cfg(feature = "gpu")]
const GPU_WARP: (&str, &str) = ("warp", concat!(include_str!("gpu_sample.wgsl"), "\n", include_str!("gpu_warp.wgsl")));
#[cfg(feature = "gpu")]
const GPU_DISTANCE: (&str, &str) = ("distance", concat!(include_str!("gpu_sample.wgsl"), "\n", include_str!("gpu_distance.wgsl")));
#[cfg(feature = "gpu")]
const GPU_BLUR: (&str, &str) = ("blur", concat!(include_str!("gpu_sample.wgsl"), "\n", include_str!("gpu_blur.wgsl")));
#[cfg(feature = "gpu")]
const GPU_MBLUR: (&str, &str) = ("motionblur", concat!(include_str!("gpu_sample.wgsl"), "\n", include_str!("gpu_mblur.wgsl")));
/// The resident id of a node-constant table built on the CPU (filter taps and offsets, gradient
/// map and curve tables), bound for one dispatch and released after it: no graph node has it.
#[cfg(feature = "gpu")]
const GPU_TABLE: usize = usize::MAX;


/// The colour kernels: the prelude (added by gpu.rs), gpu_color_common.wgsl, then each one's file.
#[cfg(feature = "gpu")]
const GPU_GRADIENT: (&str, &str) = ("gradient map", concat!(include_str!("gpu_color_common.wgsl"), include_str!("gpu_gradient.wgsl")));
#[cfg(feature = "gpu")]
const GPU_HSL: (&str, &str) = ("hsl", concat!(include_str!("gpu_color_common.wgsl"), include_str!("gpu_hsl.wgsl")));
#[cfg(feature = "gpu")]
const GPU_EMBOSS: (&str, &str) = ("emboss", concat!(include_str!("gpu_color_common.wgsl"), include_str!("gpu_emboss.wgsl")));
#[cfg(feature = "gpu")]
const GPU_DYNGRAD: (&str, &str) = ("dynamic gradient", concat!(include_str!("gpu_color_common.wgsl"), include_str!("gpu_dyngrad.wgsl")));
#[cfg(feature = "gpu")]
const GPU_CURVE: (&str, &str) = ("curve", concat!(include_str!("gpu_color_common.wgsl"), include_str!("gpu_curve.wgsl")));
pub struct Runtime<'p> {
    pub p: &'p Program,
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::Gpu>,
    pub inputs: HashMap<u32, Value>,
    /// String inputs (interface type 6) by uid: the XML default, or --set (a text node's string).
    pub strings: HashMap<u32, String>,
    /// Image inputs (interface type 5) supplied by the caller, by input uid: the op 0x20 nodes
    /// that read a graph input use them (`set_image`); unset inputs use the XML/program default.
    pub image_inputs: HashMap<u32, (std::sync::Arc<Image>, u32)>,
    pub images: Vec<Option<std::sync::Arc<Image>>>,
    pub vp: HashMap<usize, Value>,
    /// value-processor output slots (op 6's second operand) → value
    pub vp_slots: HashMap<u16, Value>,
    /// Value processors in node order: op 3's operand is an ordinal into this list.
    pub vp_nodes: Vec<usize>,
    funcs: HashMap<usize, Func>,
    /// Nodes whose op (or a feature of it) is not implemented: index → reason.
    pub unsupported: Vec<(usize, String)>,
    /// Size of the node being run, for $size in its functions.
    pub cur_size: [f32; 2],
    /// Seconds spent per opword byte0.
    pub op_time: std::collections::BTreeMap<u32, f64>,
    /// With the Vulkan backend: nodes per opcode that ran on the GPU / on the CPU.
    pub gpu_ops: std::collections::BTreeMap<u32, (usize, usize)>,
    mip_cache: HashMap<(usize, u32), std::sync::Arc<Vec<Image>>>,
    /// log2 size of every node as run (static, or from its size function).
    pub sizes: Vec<(u32, u32)>,
    /// Function variables of the node being run: its size, seed and parameter functions (and an
    /// FX-Map's tree, a pixel processor's pixels) share them, e.g. a size function sets the ROIs.
    pub node_vars: Vec<(u16, Value)>,
    /// `run_for` targets are handed here as soon as they are computed and then freed like any
    /// other image (sdrt nodes: thousands of targets would otherwise all stay in memory).
    pub sink: Option<Box<dyn FnMut(usize, &Image) + Send + Sync + 'p>>,
    /// Incremental rendering state (`crate::api::Instance`): which node results to reuse.
    pub incr: Option<Incr>,
    /// Function offsets the FX-Map evaluator decoded for the node being run (incremental
    /// rendering derives the node's graph-input reads from them).
    pub fx_trace: std::sync::Mutex<Vec<usize>>,
    /// Function offsets `func` returned for the node being run (incremental rendering).
    trace: Vec<usize>,
}

/// What a node read when it was last computed: the graph inputs (uids) and the value-processor
/// slots with the values they had.
#[derive(Clone, Debug, Default)]
pub struct NodeReads {
    pub uids: Vec<u32>,
    pub slots: Vec<(u16, Value)>,
    pub size: (u32, u32),
    pub unsupported: Vec<String>,
}

/// Cache of node results kept between renders of one graph. A node is reused when its result
/// is kept, none of the graph inputs it read changed, none of its input nodes was recomputed and
/// the value-processor slots it read hold the same values; anything else is recomputed.
#[derive(Default)]
pub struct Incr {
    /// graph inputs (uids) whose value differs from the previous render
    pub changed: std::collections::HashSet<u32>,
    /// per node: its result is kept (in `images` and/or resident on the GPU)
    pub cached: Vec<bool>,
    pub reads: Vec<NodeReads>,
    pub bytes: Vec<usize>,
    /// seconds the node took when last computed (with `bytes`: what keeping it saves)
    pub cost: Vec<f64>,
    pub budget: usize,
    pub used: usize,
    /// per function offset: the input uids and value-processor slots it reads
    func_reads: HashMap<usize, (Vec<u32>, Vec<u16>)>,
    pub recomputed: usize,
    pub reused: usize,
}

/// Environment of a node's parameter functions (evaluated once per node).
pub struct NodeEnv<'a> {
    pub rt: &'a Runtime<'a>,
    pub state: u32,
    pub pos: [f32; 2],
    pub number: f32,
    /// The node's size in pixels ($size; $sizelog2 from it).
    pub size: [f32; 2],
    /// A value processor's writes: (output slot, value).
    pub writes: Vec<(u16, Value)>,
    /// A value processor's image inputs (it samples them like a pixel processor) and their formats.
    pub srcs: Vec<std::sync::Arc<Image>>,
    pub fmts: Vec<u32>,
    /// A value processor: rand draws like a pixel processor's at $pos (0, 0) — Jenkins hash of
    /// (seed, 0, 0), then the pixel processor LCG (plaid-fabric-01 node 158, seeds 0 and 12345).
    pub vp: bool,
    /// The last sampler's filter operand (bit 1: edges clamped).
    pub filter: u16,
}

/// System variables every function context shares: 1 $size (pixels), 3 $sizelog2.
pub fn size_sysvar(index: u16, size: [f32; 2]) -> Option<Value> {
    match index {
        1 => Some(Value::float(&size)),
        3 => Some(Value::float(&[size[0].log2(), size[1].log2()])),
        _ => None,
    }
}

impl Env for NodeEnv<'_> {
    fn input(&self, uid: u32) -> Value {
        self.rt.input(uid)
    }
    fn sysvar(&self, index: u16) -> Value {
        if let Some(v) = size_sysvar(index, self.size) {
            return v;
        }
        match index {
            8 => Value::float(&self.pos),
            10 => Value::f1(self.number),
            _ => Value::f1(0.0),
        }
    }
    fn sample(&mut self, input: usize, pos: [f32; 2], color: bool, nearest: bool) -> [f32; 4] {
        // a value processor samples its inputs as a pixel processor does (raw_copper VP 241 reads a
        // 1×1 bounding box; returning 0 made VP 242 write NaN and PP 243 sample nonsense)
        let Some(img) = self.srcs.get(input) else { return [0.0; 4] };
        let wrap = if self.filter & 2 != 0 { (false, false) } else { (true, true) };
        // on the CPU like FX-Map functions: exact float weights (SDRT_VPSAMPLE=tex: texture unit)
        let s = if nearest { img.nearest(pos[0], pos[1], wrap) }
            else if crate::envc::var("SDRT_VPSAMPLE").as_deref() == Ok("tex") { img.bilinear_or(pos[0], pos[1], wrap, None, self.fmts.get(input).copied().unwrap_or(3)) }
            else { img.bilinear_exact(pos[0], pos[1], wrap) };
        if color || !img.color { s } else { let l = (s[0] + s[1] + s[2]) / 3.0; [l, l, l, 1.0] }
    }
    fn sample_filter(&mut self, filter: u16) {
        self.filter = filter;
    }
    fn vp_value(&self, node: usize) -> Value {
        self.rt.vp_slot(node)
    }
    fn write(&mut self, slot: u16, v: Value) {
        self.writes.push((slot, v));
    }
    fn rand(&mut self) -> f32 {
        if self.vp {
            self.state = (QUAD_A.wrapping_mul(self.state).wrapping_add(QUAD_C)) & M24;
            return (self.state >> 1) as f32 / 8388608.0;
        }
        self.state = (NODE_A.wrapping_mul(self.state).wrapping_add(NODE_C)) & M24;
        self.state as f32 / 16777216.0
    }
    fn cpu_trig(&self) -> bool {
        crate::env_once!("SDRT_LIBMTRIG").is_err()
    }
}

/// Environment of a pixel processor's function, per pixel.
struct PixEnv<'a> {
    rt: &'a Runtime<'a>,
    size: [f32; 2],
    srcs: &'a [&'a Image],
    fmts: &'a [u32],
    wrap: (bool, bool),
    pos: [f32; 2],
    state: u32,
    written: Option<Value>,
    /// the last sampler's filter operand (bit 1: edges clamped)
    filter: u16,
}

impl Env for PixEnv<'_> {
    fn input(&self, uid: u32) -> Value {
        self.rt.input(uid)
    }
    fn sysvar(&self, index: u16) -> Value {
        if let Some(v) = size_sysvar(index, self.size) {
            return v;
        }
        match index {
            8 => Value::float(&self.pos),
            _ => Value::f1(0.0),
        }
    }
    fn sample(&mut self, input: usize, pos: [f32; 2], color: bool, nearest: bool) -> [f32; 4] {
        let Some(img) = self.srcs.get(input) else { return [0.0; 4] };
        // bilinear: the texture unit (coupled 1/256 weights; UNORM inputs in 16-bit integers)
        let wrap = if self.filter & 2 != 0 { (false, false) } else { self.wrap };
        let s = if nearest { img.nearest(pos[0], pos[1], wrap) } else { img.bilinear_or(pos[0], pos[1], wrap, None, self.fmts.get(input).copied().unwrap_or(3)) };
        if color || !img.color {
            s
        } else {
            // a color input: (r + g + b)/3, measured (lab/py/lumprobe.py)
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
    fn rand(&mut self) -> f32 {
        self.state = (QUAD_A.wrapping_mul(self.state).wrapping_add(QUAD_C)) & M24;
        (self.state >> 1) as f32 / 8388608.0
    }
    fn write(&mut self, _slot: u16, v: Value) {
        self.written = Some(v);
    }
}

/// A pixel processor's rand state at a pixel: Bob Jenkins' one-at-a-time hash of three 32-bit words,
/// the node seed and the float bits of $pos.x and $pos.y, masked to 24 bits (scratch/pp2/oat.py,
/// exact on every measured state, GPU and CPU engines; the CPU engine computes sides below 16 px as 16).
fn pixel_seed(seed: u32, pos: [f32; 2]) -> u32 {
    let mut h: u32 = 0;
    for w in [seed, pos[0].to_bits(), pos[1].to_bits()] {
        h = h.wrapping_add(w);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);
    h & M24
}

fn floats(ws: &[u32]) -> Vec<f32> {
    ws.iter().map(|&w| f32::from_bits(w)).collect()
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

impl<'p> Runtime<'p> {
    pub fn new(p: &'p Program) -> Runtime<'p> {
        Runtime {
            p,
            #[cfg(feature = "gpu")]
            gpu: None,
            inputs: p.defaults.clone(),
            strings: HashMap::new(),
            image_inputs: HashMap::new(),
            images: vec![None; p.nodes.len()],
            vp: HashMap::new(),
            vp_slots: HashMap::new(),
            vp_nodes: p.nodes.iter().filter(|n| n.op == 0x28 && n.flags >> 16 != 0).map(|n| n.index).collect(),
            funcs: HashMap::new(),
            unsupported: vec![],
            cur_size: [256.0, 256.0],
            op_time: Default::default(),
            gpu_ops: Default::default(),
            mip_cache: HashMap::new(),
            sizes: p.nodes.iter().map(|n| n.log2).collect(),
            node_vars: Vec::new(),
            sink: None,
            incr: None,
            fx_trace: std::sync::Mutex::new(Vec::new()),
            trace: Vec::new(),
        }
    }

    /// Take the GPU context back with its resident images (incremental rendering keeps them).
    #[cfg(feature = "gpu")]
    pub fn take_gpu(&mut self) -> Option<crate::gpu::Gpu> {
        if let Some(g) = &mut self.gpu {
            g.flush();
        }
        self.gpu.take()
    }

    /// Use an existing GPU context (its pipelines stay cached across runtimes).
    #[cfg(feature = "gpu")]
    pub fn attach_gpu(&mut self, gpu: crate::gpu::Gpu) {
        self.gpu = Some(gpu);
    }
    /// Take the GPU context back, its graph images released.
    #[cfg(feature = "gpu")]
    pub fn detach_gpu(&mut self) -> Option<crate::gpu::Gpu> {
        if let Some(g) = &mut self.gpu {
            g.reset();
        }
        self.gpu.take()
    }
    pub fn enable_gpu(&mut self) -> Result<(), String> {
        #[cfg(feature = "gpu")]
        { self.gpu = Some(crate::gpu::Gpu::new()?); Ok(()) }
        #[cfg(not(feature = "gpu"))]
        { Err("Vulkan backend requires cargo build --features gpu".into()) }
    }

    pub fn input(&self, uid: u32) -> Value {
        self.inputs.get(&uid).copied().unwrap_or(Value::int(&[0]))
    }

    pub fn func(&mut self, off: usize) -> Func {
        if self.incr.is_some() {
            self.trace.push(off);
        }
        self.funcs.entry(off).or_insert_with(|| Func::decode(self.p, off)).clone()
    }

    fn seed(&mut self, n: &Node) -> u32 {
        match &n.seed {
            Slot::Const(w) => w[0],
            Slot::Fn(off) => {
                let f = self.func(*off);
                let mut vars = std::mem::take(&mut self.node_vars);
                let mut env = NodeEnv { rt: self, state: 0, pos: [0.5, 0.5], number: 0.0, size: [n.size().0 as f32, n.size().1 as f32], writes: Vec::new(), srcs: Vec::new(), fmts: Vec::new(), vp: false, filter: 0 };
                let v = Eval::run_with_vars(&f, &mut env, &mut vars).iv()[0] as u32;
                self.node_vars = vars;
                v
            }
            _ => 0,
        }
    }

    /// A node-level parameter: constant, or its function evaluated with the node's random stream.
    pub fn param(&mut self, n: &Node, name: &str, seed: u32) -> Option<Vec<f32>> {
        self.slot(n.param(name), seed)
    }

    pub fn slot(&mut self, s: &Slot, seed: u32) -> Option<Vec<f32>> {
        match s {
            Slot::Const(w) => Some(floats(w)),
            Slot::Fn(off) => {
                let f = self.func(*off);
                let size = self.cur_size;
                let mut vars = std::mem::take(&mut self.node_vars);
                let mut env = NodeEnv { rt: self, state: seed & M24, pos: [0.5, 0.5], number: 0.0, size, writes: Vec::new(), srcs: Vec::new(), fmts: Vec::new(), vp: false, filter: 0 };
                let v = Eval::run_with_vars(&f, &mut env, &mut vars);
                self.node_vars = vars;
                // always four components: a scalar (or nothing) where a vector is read broadcasts
                Some(if v.n() <= 1 { vec![v.fv()[0]; 4] } else { v.fv().to_vec() })
            }
            _ => None,
        }
    }

    /// Op 3 reads value-processor output slot k: the value a value processor wrote with
    /// write(value, k) (pine_wood_panel_v2: node 885 writes slot 12, read by FX-Map 886; the slots
    /// are not the value processors' ordinals, which coincided on cracks-cliff blend 73).
    pub fn vp_slot(&self, k: usize) -> Value {
        self.vp_slots.get(&(k as u16)).copied().unwrap_or(Value::f1(0.0))
    }

    /// The pixelsize slot (a scale of the intensity's pixel distances, per axis): constant,
    /// function (Adobe's materials use 0.1·2048/$size) or (1, 1).
    fn pixelsize(&mut self, n: &Node, seed: u32) -> (f32, f32) {
        let s = n.pixelsize.clone();
        match self.slot(&s, seed) {
            Some(v) => (v[0], v.get(1).copied().unwrap_or(v[0])),
            None => (1.0, 1.0),
        }
    }

    fn unsupported(&mut self, i: usize, why: impl Into<String>) {
        self.unsupported.push((i, why.into()));
    }

    fn img(&self, i: usize) -> &Image {
        self.images[i].as_ref().expect("input computed")
    }

    /// Input i at the node's size: borrowed when it already has it.
    /// (A shared handle: holding it does not borrow the runtime, so parameters can still be read.)
    fn input_at(&self, i: usize, w: usize, h: usize) -> std::sync::Arc<Image> {
        let im = self.images[i].as_ref().expect("input computed");
        if (im.w, im.h) == (w, h) { im.clone() } else { std::sync::Arc::new(im.resized(w, h)) }
    }

    /// Input i at the node's size, resampled in its texture format (UNORM formats ≤ 1 interpolate
    /// in 16-bit integers with coupled 1/256 weights, as the texture unit).
    fn input_at_fmt(&self, i: usize, w: usize, h: usize) -> std::sync::Arc<Image> {
        let im = self.images[i].as_ref().expect("input computed");
        if (im.w, im.h) == (w, h) { im.clone() } else { std::sync::Arc::new(im.resized_fmt(w, h, self.tex_format(i))) }
    }

    /// Run every node. Images no longer needed are dropped unless `keep_all`.
    pub fn run(&mut self, keep_all: bool) {
        self.run_inner(keep_all, None)
    }

    /// Compute only what `targets` depend on and keep only their images (memory stays bounded on
    /// big graphs: `run(true)` keeps every node of the program).
    pub fn run_for(&mut self, targets: &[usize]) {
        self.run_inner(false, Some(targets))
    }

    fn run_inner(&mut self, keep_all: bool, targets: Option<&[usize]>) {
        #[cfg(feature="gpu")]
        if let Some(gpu)=&mut self.gpu { if self.incr.is_some() { gpu.flush(); } else { gpu.reset(); } }
        let n = self.p.nodes.len();
        let outs: Vec<usize> = match targets {
            Some(t) => t.to_vec(),
            None => self.p.outputs.iter().map(|o| o.node).collect(),
        };
        let node_srcs = |node: &Node| -> Vec<usize> {
            node.inputs.iter().copied().chain(node.params.iter().filter_map(|(_, s)| if let Slot::Input(k) = s { Some(*k) } else { None })).collect()
        };
        let is_vp = |nd: &Node| nd.op == 0x28 && nd.flags >> 16 != 0;
        // incremental: nodes that may need recomputing (not kept, a changed input read, an input
        // node that may be recomputed, or value-processor slots read). The inputs of the others
        // are not needed: their kept results are reused.
        let maybe_dirty: Option<Vec<bool>> = self.incr.as_mut().map(|inc| {
            inc.cached.resize(n, false);
            inc.reads.resize(n, NodeReads::default());
            inc.bytes.resize(n, 0);
            inc.cost.resize(n, 0.0);
            let mut md = vec![false; n];
            for (k, nd) in self.p.nodes.iter().enumerate() {
                let r = &inc.reads[k];
                md[k] = !inc.cached[k] || is_vp(nd) || !r.slots.is_empty() || r.uids.iter().any(|u| inc.changed.contains(u))
                    || node_srcs(nd).iter().any(|&j| j >= k || md[j]);
            }
            md
        });
        // only nodes the outputs depend on (sbsrender skips the rest too)
        let mut needed = vec![keep_all; n];
        let mut stack: Vec<usize> = outs.clone();
        // value processors are read by functions (op 3) from anywhere: always run them, and what
        // they sample (raw_copper VP 241 reads node 240, which nothing else needed)
        stack.extend(self.p.nodes.iter().filter(|nd| nd.op == 0x28 && nd.flags >> 16 != 0).map(|nd| nd.index));
        while let Some(k) = stack.pop() {
            if k >= n || needed[k] && !keep_all {
                continue;
            }
            needed[k] = true;
            if maybe_dirty.as_ref().is_some_and(|md| !md[k]) {
                continue;
            }
            let node = &self.p.nodes[k];
            stack.extend(node.inputs.iter().copied().filter(|&j| j < n && !needed[j]));
            for (_, s) in &node.params {
                if let Slot::Input(j) = s {
                    if *j < n && !needed[*j] {
                        stack.push(*j);
                    }
                }
            }
        }
        // value processors are read by functions (op 3) from anywhere: always run them
        for (k, node) in self.p.nodes.iter().enumerate() {
            if node.op == 0x28 && node.flags >> 16 != 0 {
                needed[k] = true;
            }
        }
        let mut is_out = vec![false; n];
        for &k in &outs { if k < n { is_out[k] = true; } }
        // last consumer of each image among the nodes that run
        let mut last_use = vec![usize::MAX; n];
        for node in self.p.nodes.iter().filter(|nd| needed[nd.index]) {
            for i in node.inputs.iter().copied().chain(node.params.iter().filter_map(|(_,s)|if let Slot::Input(k)=s {Some(*k)} else {None})) {
                if i<n {last_use[i]=node.index;}
            }
        }
        let mut recomputed = vec![false; n];
        for i in 0..n {
            if !needed[i] {
                continue;
            }
            if let Some(inc) = &mut self.incr {
                let nd = &self.p.nodes[i];
                let r = &inc.reads[i];
                let reuse = inc.cached[i]
                    && !is_vp(nd)
                    && !r.uids.iter().any(|u| inc.changed.contains(u))
                    && !node_srcs(nd).iter().any(|&j| j >= i || recomputed[j])
                    && r.slots.iter().all(|(s, v)| self.vp_slots.get(s).copied().unwrap_or(Value::f1(0.0)) == *v);
                if reuse {
                    inc.reused += 1;
                    self.sizes[i] = r.size;
                    for u in r.unsupported.clone() { self.unsupported.push((i, u)); }
                    #[cfg(feature="gpu")]
                    if is_out[i] { self.materialize(i); }
                    continue;
                }
                inc.recomputed += 1;
                recomputed[i] = true;
                // drop the stale result
                inc.used -= std::mem::take(&mut inc.bytes[i]);
                inc.cached[i] = false;
                self.images[i] = None;
                #[cfg(feature="gpu")]
                if let Some(g)=&mut self.gpu { g.release(i); }
                self.trace.clear();
                self.fx_trace.lock().unwrap_or_else(|e| e.into_inner()).clear();
            }
            let unsupported_before = self.unsupported.len();
            let node = self.p.nodes[i].clone();
            // an unconnected pixel-processor input (0xFFFFFFFF, v9: kachelboden 364/382) samples as
            // transparent black; any other input at or after the node is unsupported
            if node.inputs.iter().any(|&k| k >= i && !(k == 0xFFFF_FFFF && node.op & 0xFE == 0x28)) {
                self.unsupported(i, "input after the node");
                self.images[i] = Some(std::sync::Arc::new(Image::new(1, 1, node.color, [0.0; 4])));
                continue;
            }
            let mut node = node;
            self.node_vars.clear();
            if let Slot::Fn(off) = node.size {
                // the size function gives log2 (width, height), e.g. from $outputsize
                let f = self.func(off);
                let mut vars = Vec::new();
                // its rand draws from the node LCG started at the node's seed (lab/py/sizerand.py;
                // Adobe's noise pixel processors pick their random offset there)
                let seed = self.seed(&node);
                let mut env = NodeEnv { rt: self, state: seed & M24, pos: [0.5, 0.5], number: 0.0, size: [256.0, 256.0], writes: Vec::new(), srcs: Vec::new(), fmts: Vec::new(), vp: false, filter: 0 };
                let v = Eval::run_with_vars(&f, &mut env, &mut vars).iv();
                self.node_vars = vars;
                node.log2 = (v[0].clamp(0, 13) as u32, v[1].clamp(0, 13) as u32);
                self.sizes[i] = node.log2;
            }
            let t0 = std::time::Instant::now();
            #[cfg(feature="gpu")]
            let resident=self.try_gpu(&node);
            #[cfg(feature="gpu")]
            if self.gpu.is_some() {
                let e=self.gpu_ops.entry(node.op).or_default();
                if resident {e.0+=1} else {e.1+=1}
            }
            #[cfg(not(feature="gpu"))]
            let resident=false;
            if !resident {
                #[cfg(feature="gpu")]
                {
                    self.materialize_for_node(&node);
                }
                let mut img=self.exec(&node);
                img.quantize(node.format());
                self.images[i]=Some(std::sync::Arc::new(img));
            }
            #[cfg(feature="gpu")]
            if keep_all || is_out[i] { self.materialize(i); }
            let dt = t0.elapsed().as_secs_f64();
            // op | color bit: the timing key keeps the colour flag, so aggregate the opcode
            // separately (SDRT_TIME prints one line per key; the opcode totals are what matter).
            *self.op_time.entry(node.op).or_default() += dt;
            if dt > 0.25 && crate::envc::var("SDRT_TIME").is_ok_and(|v| v == "2") {
                let ins: Vec<String> = node.inputs.iter().map(|&k| self.images[k].as_ref().map_or("-".into(), |m| format!("{}x{}", m.w, m.h))).collect();
                eprintln!("slow: node {} op {:#04x} {}x{} {:.2}s inputs {:?}", i, node.op, node.size().0, node.size().1, dt, ins);
            }
            let streamed = self.sink.is_some() && is_out[i];
            if self.incr.is_some() {
                self.record_incr(&node, unsupported_before, dt, &last_use, &is_out);
            }
            if streamed {
                (self.sink.as_mut().unwrap())(i, self.images[i].as_ref().expect("export materialized"));
            }
            if streamed && last_use[i] == usize::MAX && !keep_all {
                self.images[i]=None;
                #[cfg(feature="gpu")]
                if let Some(g)=&mut self.gpu { g.release(i); }
            }
            if !keep_all {
                for k in node.inputs.iter().copied().chain(node.params.iter().filter_map(|(_,s)|if let Slot::Input(k)=s {Some(*k)} else {None})) {
                    if k < n && last_use[k] == i && (!is_out[k] || self.sink.is_some()) && !self.incr.as_ref().is_some_and(|c| c.cached[k]) {
                        self.images[k] = None;
                        #[cfg(feature="gpu")]
                        if let Some(g)=&mut self.gpu { g.release(k); }
                        self.mip_cache.retain(|&(id, _), _| id != k);
                    }
                }
            }
        }
        if let Some(inc) = &mut self.incr {
            let mut stale = vec![false; n];
            for (i, nd) in self.p.nodes.iter().enumerate() {
                if needed[i] {
                    continue;
                }
                let r = &inc.reads[i];
                stale[i] = r.uids.iter().any(|u| inc.changed.contains(u))
                    || node_srcs(nd).iter().any(|&j| j < n && (j >= i || stale[j] || recomputed[j]));
                if stale[i] && inc.cached[i] {
                    inc.cached[i] = false;
                    inc.used -= std::mem::take(&mut inc.bytes[i]);
                }
            }
        }
        #[cfg(feature="gpu")]
        if let Some(gpu) = &mut self.gpu {
            gpu.flush();
        }
    }

    /// Incremental rendering: note what node `node` read (the functions it evaluated, its bitmap
    /// or text input) and keep its result while the cache budget allows.
    fn record_incr(&mut self, node: &Node, unsupported_before: usize, dt: f64, last_use: &[usize], is_out: &[bool]) {
        let i = node.index;
        let mut offs = std::mem::take(&mut self.trace);
        offs.append(&mut self.fx_trace.lock().unwrap_or_else(|e| e.into_inner()));
        offs.sort_unstable();
        offs.dedup();
        let (mut uids, mut slots) = (vec![], vec![]);
        for off in offs {
            let p = self.p;
            let funcs = &self.funcs;
            let inc = self.incr.as_mut().unwrap();
            let (u, s) = inc.func_reads.entry(off).or_insert_with(|| match funcs.get(&off) { Some(f) => f.graph_reads(), None => Func::decode(p, off).graph_reads() });
            uids.extend_from_slice(u);
            slots.extend_from_slice(s);
        }
        // graph-input bitmaps (op 0x20 with b3 0) and text from a string input read their uid directly
        if (node.op == 0x20 && node.b3 == 0) || (node.op == 0x22 && node.flags & 3 == 2) {
            uids.push(node.data as u32);
        }
        uids.sort_unstable();
        uids.dedup();
        slots.sort_unstable();
        slots.dedup();
        let slots: Vec<(u16, Value)> = slots.into_iter().map(|s| (s, self.vp_slot(s as usize))).collect();
        let unsupported = self.unsupported[unsupported_before..].iter().map(|(_, w)| w.clone()).collect();
        #[cfg(feature = "gpu")]
        let dims = self.gpu.as_ref().and_then(|g| g.size(i)).or_else(|| self.images[i].as_ref().map(|m| (m.w, m.h)));
        #[cfg(not(feature = "gpu"))]
        let dims = self.images[i].as_ref().map(|m| (m.w, m.h));
        let size = self.sizes[i];
        let inc = self.incr.as_mut().unwrap();
        inc.reads[i] = NodeReads { uids, slots, size, unsupported };
        inc.cost[i] = dt;
        let Some((w, h)) = dims else { return };
        let bytes = w * h * 16;
        if bytes > inc.budget {
            return;
        }
        // over budget: evict kept results that save less time per byte than this one
        let worth = |c: f64, b: usize| c / b.max(1) as f64;
        let mine = worth(dt, bytes);
        let mut evict = vec![];
        while inc.used + bytes > inc.budget {
            let victim = (0..inc.cached.len())
                .filter(|&k| inc.cached[k] && k != i && !evict.contains(&k))
                .min_by(|&a, &b| worth(inc.cost[a], inc.bytes[a]).total_cmp(&worth(inc.cost[b], inc.bytes[b])));
            match victim {
                Some(k) if worth(inc.cost[k], inc.bytes[k]) < mine => {
                    inc.used -= inc.bytes[k];
                    evict.push(k);
                }
                _ => break,
            }
        }
        if inc.used + bytes > inc.budget {
            // not kept: give the evictions back
            for k in evict {
                inc.used += inc.bytes[k];
            }
            return;
        }
        for &k in &evict {
            inc.cached[k] = false;
            inc.bytes[k] = 0;
        }
        inc.used += bytes;
        inc.bytes[i] = bytes;
        inc.cached[i] = true;
        // free evicted results this render no longer reads (the others are freed after their
        // last use; outputs at the end)
        for k in evict {
            if !is_out[k] && (last_use[k] < i || last_use[k] == usize::MAX) {
                self.images[k] = None;
                #[cfg(feature = "gpu")]
                if let Some(g) = &mut self.gpu { g.release(k); }
            }
        }
    }

    /// Incremental rendering, after the outputs were read: free every result that is not kept.
    pub fn incr_finish(&mut self) {
        let Some(inc) = &self.incr else { return };
        for k in 0..self.images.len() {
            if !inc.cached.get(k).copied().unwrap_or(false) {
                self.images[k] = None;
                #[cfg(feature = "gpu")]
                if let Some(g) = &mut self.gpu { g.release(k); }
            }
        }
        #[cfg(feature = "gpu")]
        if let Some(g) = &mut self.gpu { g.flush(); }
    }

    /// The transformation node (op 0x04) on the GPU, as a chain of gpu_xf.wgsl passes that
    /// mirrors the CPU path ("transformation" below) step by step:
    ///
    /// - auto-mip bilinear signed permutations (the dihedral path): per-axis halvings in the
    ///   node's format, then per-axis 2x bilinear upsamples in the source's format, then the
    ///   texel permutation (identity: a copy);
    /// - otherwise the mip level the CPU picks: levels >= 1 are rebuilt by halvings in the
    ///   transformation's mip format, then sampled (nearest or 1/256 bilinear, matte or
    ///   clamped border).
    ///
    /// CPU fallbacks: a missing input, the SDRT_XF*/SDRT_DS*/SDRT_MIPROUND experiment switches,
    /// non-power-of-two sizes where a resize or mip chain is needed, non-finite or huge
    /// coordinates, and images over one storage binding. A fallback after the parameters were
    /// evaluated restores the node variables, so the CPU re-evaluates the seed and parameter
    /// functions from the same state.
    #[cfg(feature = "gpu")]
    fn try_gpu_transform(&mut self, n: &Node) -> bool {
        use crate::gpu::{Pass, Src};
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let early = |why: &str| {
            if log {
                eprintln!("GPU transform node {}: CPU fallback ({why})", n.index);
            }
            false
        };
        let k = n.inputs[0];
        if k >= self.p.nodes.len() {
            return early("missing input");
        }
        const SWITCHES: [&str; 14] = [
            "SDRT_XFIDENT", "SDRT_XFAXES", "SDRT_XFUPF", "SDRT_XFMIPF", "SDRT_XFMIPBIAS", "SDRT_XFUPSHIFT",
            "SDRT_XFUPS", "SDRT_XFUNORM", "SDRT_XFBORDER", "SDRT_DSROUND", "SDRT_DSINT", "SDRT_MIPROUND",
            "SDRT_XFEDGE",
            "SDRT_XFMIP8",
        ];
        if SWITCHES.iter().any(|s| crate::envc::var(s).is_ok()) {
            return early("experiment switch");
        }
        let (resident, cpu_img) = (self.gpu.as_ref().unwrap().size(k), self.images.get(k).and_then(|i| i.as_ref()));
        let Some((sw, sh)) = resident.or_else(|| cpu_img.map(|im| (im.w, im.h))) else { return early("input not computed") };
        let src_color = self.gpu.as_ref().unwrap().color(k).or_else(|| cpu_img.map(|im| im.color)).unwrap();
        let (w, h) = n.size();
        if !self.gpu.as_ref().unwrap().fits(sw, sh) || !self.gpu.as_ref().unwrap().fits(w, h) {
            return early("image exceeds one GPU storage binding");
        }
        // Evaluate the seed and the parameters exactly as exec() does before "transformation".
        let saved_vars = self.node_vars.clone();
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let matrix = self.param(n, "matrix", seed).unwrap_or(vec![1.0, 0.0, 0.0, 1.0]);
        let offset = self.param(n, "offset", seed).unwrap_or(vec![0.0, 0.0]);
        let matte = self.param(n, "mattecolor", seed).unwrap_or(vec![0.0, 0.0, 0.0, 0.0]);
        let reject = |rt: &mut Self, why: &str| {
            rt.node_vars = saved_vars.clone();
            early(why)
        };
        let tiling = n.b3 & 3;
        let wrap = (tiling & 1 != 0, tiling & 2 != 0);
        let bilinear = n.flags & 0x20 != 0;
        let manual = match n.flags & 0x1F {
            0x1F => None,
            0x1E => Some(self.param(n, "miplevel", seed).map_or(0.0, |v| v[0])),
            m => Some(m as f32),
        };
        let (m, off) = (&matrix[..4], &offset[..2]);
        if crate::envc::var("SDRT_STRICT_TRACE").is_ok() {
            eprintln!("STRICT {{\"k\":{},\"op\":4,\"input\":{k},\"sfmt\":{},\"fmt\":{},\"sw\":{sw},\"sh\":{sh},\"w\":{w},\"h\":{h},\"flags\":{},\"tiling\":{tiling},\"matrix\":{m:?},\"offset\":{off:?},\"manual\":{}}}", n.index, self.p.nodes[k].format(), n.format(), n.flags, manual.map_or("null".into(), |v| v.to_string()));
        }
        let identity = m == [1.0, 0.0, 0.0, 1.0] && off == [0.0, 0.0];
        let dihedral = off == [0.0, 0.0]
            && ((m[0].abs() == 1.0 && m[1] == 0.0 && m[2] == 0.0 && m[3].abs() == 1.0)
                || (m[0] == 0.0 && m[1].abs() == 1.0 && m[2].abs() == 1.0 && m[3] == 0.0));
        let pow2 = sw.is_power_of_two() && sh.is_power_of_two();
        // A pass config: the operation, its flags, the matrix/offset/matte and the input descriptor.
        let pass = |op: u32, flags: u32, input: (usize, usize), in_fmt: u32, out: (usize, usize), color: bool, format: u32| {
            let mut config = [0u32; 40];
            // The Config layout: header[0..4], values[4..8], extra[8..12], selectors[12..16],
            // options[16..20], parameters[20..40] (five vec4<f32>).
            config[0] = op;
            config[3] = flags;
            config[20..24].copy_from_slice(&[m[0], m[1], m[2], m[3]].map(f32::to_bits));
            config[24..28].copy_from_slice(&[off[0], off[1], 0.0, 0.0].map(f32::to_bits));
            let matte_c = if n.color { [matte[0], matte[1], matte[2], matte[3]] } else { Image::gray(matte[0]) };
            config[8..12].copy_from_slice(&matte_c.map(f32::to_bits));
            config[12..16].copy_from_slice(&[input.0 as u32, input.1 as u32, in_fmt, u32::from(src_color)]);
            Pass { config, w: out.0, h: out.1, color, format, a: Src::None, b: Src::None }
        };
        let axes = |x: bool, y: bool| u32::from(x) << 8 | u32::from(y) << 9;
        let wrap_bits = u32::from(wrap.0) | u32::from(wrap.1) << 1;
        let mut passes = Vec::new();
        let mut level = 0usize;
        let path;
        if dihedral && bilinear && manual.is_none() {
            // The CPU's dihedral path: per-axis resize to (tw, th), then a texel permutation.
            let (tw, th) = if m[0] == 0.0 { (h, w) } else { (w, h) };
            if (sw, sh) != (tw, th) && !pow2 {
                return reject(self, "non-power-of-two resize");
            }
            let (mut cw, mut ch) = (sw, sh);
            let mut sample_fmt = self.p.nodes[k].format();
            // halvings stored in the transformation's own format
            while cw > tw || ch > th {
                let (ax, ay) = (cw > tw, ch > th);
                let next = (cw >> usize::from(ax), ch >> usize::from(ay));
                passes.push(pass(2, axes(ax, ay), (cw, ch), sample_fmt, next, src_color, n.format()));
                sample_fmt = n.format();
                (cw, ch) = next;
            }
            // 2x upsamples interpolated in the source's format, unquantized
            while cw < tw || ch < th {
                let (ax, ay) = (cw < tw, ch < th);
                let next = (cw << usize::from(ax), ch << usize::from(ay));
                passes.push(pass(3, wrap_bits | axes(ax, ay), (cw, ch), self.p.nodes[k].format(), next, true, 3));
                (cw, ch) = next;
            }
            passes.push(pass(1, 0, (cw, ch), 3, (w, h), n.color, n.format()));
            path = "dihedral";
        } else {
            if m.iter().chain(off).any(|v| !v.is_finite()) {
                return reject(self, "non-finite matrix or offset");
            }
            // |u| and |v| bound in source texels: beyond 2^24 the i32 texel index (Rust uses
            // i64) and floor() of an imprecise coordinate are not worth reproducing.
            let reach = |a: f32, b: f32, o: f32, size: usize| (0.5 * (a.abs() + b.abs()) + 0.5 + o.abs()) * size as f32;
            if reach(m[0], m[2], off[0], sw) >= 16_777_216.0 || reach(m[1], m[3], off[1], sh) >= 16_777_216.0 {
                return reject(self, "coordinates out of range");
            }
            // The level the CPU picks: row norms in output pixel steps, nearest level, +1/2048.
            let du = ((m[0] * sw as f32 / w as f32).powi(2) + (m[2] * sw as f32 / h as f32).powi(2)).sqrt();
            let dv = ((m[1] * sh as f32 / w as f32).powi(2) + (m[3] * sh as f32 / h as f32).powi(2)).sqrt();
            let lod_raw = du.max(dv).log2();
            level = match manual {
                Some(l) => l.max(0.0).round() as usize,
                None => (lod_raw.max(0.0) + 0.5 + 1.0 / 2048.0).floor() as usize,
            };
            let (mut cw, mut ch) = (sw, sh);
            // level 0 samples the source in its node's format; levels >= 1 are halvings in the
            // transformation's mip format (embedded bitmaps: float), the last one 1x1
            let mut fmt = self.p.nodes[k].format();
            if level > 0 {
                if !pow2 {
                    return reject(self, "non-power-of-two mip chain");
                }
                fmt = if self.p.nodes[k].op & 0xFE == 0x20 { 3 } else { n.format() };
                let mut sample_fmt = self.p.nodes[k].format();
                for _ in 0..level {
                    if cw == 1 && ch == 1 {
                        break;
                    }
                    let next = ((cw / 2).max(1), (ch / 2).max(1));
                    passes.push(pass(2, axes(cw > 1, ch > 1), (cw, ch), sample_fmt, next, src_color, fmt));
                    sample_fmt = fmt;
                    (cw, ch) = next;
                }
            }
            // An identity matrix with a manual level clamps to the edge texel; otherwise a
            // non-tiled edge blends with the matte colour.
            let flags = wrap_bits | u32::from(bilinear) << 2 | u32::from(identity) << 3;
            if fmt <= 1 && bilinear {
                // UNORM interpolation converts texels to 16-bit codes with Rust's saturating
                // casts and divides codes past 65535 exactly: the kernel assumes values in
                // [0, 1]. Stored UNORM texels are; a matte or an HDR 1x1 "mip level" may not be.
                let matte_used = !identity && !(wrap.0 && wrap.1);
                let matte_c = if n.color { [matte[0], matte[1], matte[2], matte[3]] } else { Image::gray(matte[0]) };
                if matte_used && !matte_c.iter().all(|v| (0.0..=1.0).contains(v)) {
                    return reject(self, "matte outside [0, 1] in UNORM interpolation");
                }
                if level > 0 && (cw, ch) == (sw, sh) && self.p.nodes[k].format() > 1 {
                    return reject(self, "float 1x1 source sampled as a UNORM mip level");
                }
            }
            passes.push(pass(0, flags, (cw, ch), fmt, (w, h), n.color, n.format()));
            path = "sample";
        }
        if passes.iter().any(|p| !self.gpu.as_ref().unwrap().fits(p.w, p.h)) {
            return reject(self, "intermediate exceeds one GPU storage binding");
        }
        // The source must be on the GPU: upload it when the CPU computed it.
        let g = self.gpu.as_mut().unwrap();
        if g.size(k).is_none() {
            g.upload(k, self.images[k].as_ref().expect("transform input computed")).expect("transform upload failed");
        }
        // each pass reads the previous one; the first reads the source node
        for (i, p) in passes.iter_mut().enumerate() {
            p.a = if i == 0 { Src::Node(k) } else { Src::Pass(i - 1) };
            p.b = p.a;
        }
        g.dispatch_passes(n.index, crate::gpu::XF, &passes).expect("Vulkan transform dispatch failed");
        if log {
            let last = passes.last().unwrap();
            eprintln!(
                "GPU transform node {}: shader {path} {} passes, level {level}, {}x{} from {sw}x{sh}, matrix {m:?} offset {off:?} flags {:#x}",
                n.index, passes.len(), w, h, last.config[3]
            );
        }
        self.images[n.index] = None;
        true
    }

    /// Width, height and colour flag of node k's image, resident on the GPU or on the CPU.
    #[cfg(feature = "gpu")]
    fn gpu_image_info(&self, k: usize) -> Option<(usize, usize, bool)> {
        if k >= self.p.nodes.len() {
            return None;
        }
        let g = self.gpu.as_ref()?;
        if let (Some((w, h)), Some(c)) = (g.size(k), g.color(k)) {
            return Some((w, h, c));
        }
        self.images.get(k)?.as_ref().map(|im| (im.w, im.h, im.color))
    }

    /// Dispatch a filter kernel chain for node n: its image inputs are uploaded when the CPU
    /// computed them, and `table` (node constants: taps, offsets) is bound as `Src::Node(GPU_TABLE)`
    /// for the duration of the submission.
    #[cfg(feature = "gpu")]
    fn gpu_filter_dispatch(&mut self, n: &Node, kernel: (&'static str, &'static str), inputs: &[usize], table: Option<Vec<[f32; 4]>>, passes: &[crate::gpu::Pass]) {
        let g = self.gpu.as_mut().unwrap();
        for &k in inputs {
            if g.size(k).is_none() {
                g.upload(k, self.images[k].as_ref().expect("filter input computed")).expect("filter upload failed");
            }
        }
        if let Some(px) = table {
            g.release(GPU_TABLE);
            g.upload(GPU_TABLE, &Image { w: px.len(), h: 1, color: true, px }).expect("filter table upload failed");
        }
        g.dispatch_passes(n.index, kernel, passes).unwrap_or_else(|e| panic!("Vulkan {} dispatch failed: {e}", kernel.0));
        // wgpu keeps the buffer alive until the submission that reads it is done
        g.release(GPU_TABLE);
        self.images[n.index] = None;
    }

    /// A filter pass reading `a` (descriptor `desc`: width, height, sampled format, colour) and `b`.
    #[cfg(feature = "gpu")]
    fn filter_pass(op: u32, flags: u32, desc: [u32; 4], out: (usize, usize), color: bool, format: u32, a: crate::gpu::Src, b: crate::gpu::Src) -> crate::gpu::Pass {
        let mut config = [0u32; 40];
        config[0] = op;
        config[3] = flags;
        config[12..16].copy_from_slice(&desc);
        crate::gpu::Pass { config, w: out.0, h: out.1, color, format, a, b }
    }

    /// The warp node (op 0x0E) on the GPU (gpu_warp.wgsl pass 0), mirroring "warp": both inputs
    /// are read at their own size; a gradient of the node's size takes backward texel differences,
    /// any other size bilinear differences one gradient texel apart; the source is sampled with
    /// "bilinear_or" in its node's format. kx, ky and 1/gw, 1/gh are computed here as the CPU does.
    /// CPU fallbacks: a missing input, SDRT_WARPBIG / SDRT_TEXFLOAT, images over one storage
    /// binding or sides over 65536 texels, non-finite node constants (after the parameters were
    /// evaluated: the node variables are restored).
    #[cfg(feature = "gpu")]
    fn try_gpu_warp(&mut self, n: &Node) -> bool {
        use crate::gpu::Src;
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let early = |why: &str| {
            if log {
                eprintln!("GPU warp node {}: CPU fallback ({why})", n.index);
            }
            false
        };
        if n.inputs.len() < 2 {
            return early("missing input");
        }
        let (ks, kg) = (n.inputs[0], n.inputs[1]);
        if ["SDRT_WARPBIG", "SDRT_TEXFLOAT"].iter().any(|s| crate::envc::var(s).is_ok()) {
            return early("experiment switch");
        }
        let (Some(si), Some(gi)) = (self.gpu_image_info(ks), self.gpu_image_info(kg)) else { return early("missing input") };
        let (w, h) = n.size();
        let g = self.gpu.as_ref().unwrap();
        if !g.fits(si.0, si.1) || !g.fits(gi.0, gi.1) || !g.fits(w, h) {
            return early("image exceeds one GPU storage binding");
        }
        if [si.0, si.1, gi.0, gi.1].iter().any(|&s| s > 65536) {
            return early("input side over 65536 texels");
        }
        // Evaluate the seed and the parameters exactly as exec() and "warp" do.
        let saved_vars = self.node_vars.clone();
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let inten = self.param(n, "intensity", seed).map_or(1.0, |v| v[0]);
        let ps = self.pixelsize(n, seed);
        let (gw, gh) = (gi.0, gi.1);
        let (kx, ky) = (inten * gw as f32 / 256.0 * ps.0, inten * gh as f32 / 256.0 * ps.1);
        let (igw, igh) = (1.0 / gw as f32, 1.0 / gh as f32);
        if !(kx.is_finite() && ky.is_finite()) {
            self.node_vars = saved_vars;
            return early("non-finite intensity or pixelsize");
        }
        let wrap = n.b3 & 3;
        let same = gw == w && gh == h;
        let (fmt, gfmt) = (self.tex_format(ks), self.tex_format(kg));
        let flags = wrap | u32::from(same) << 2 | u32::from(gi.2) << 3;
        let mut pass = Self::filter_pass(0, flags, [si.0 as u32, si.1 as u32, fmt, u32::from(si.2)], (w, h), n.color, n.format(), Src::Node(ks), Src::Node(kg));
        pass.config[4..8].copy_from_slice(&[gw as u32, gh as u32, gfmt, u32::from(gi.2)]);
        pass.config[20..24].copy_from_slice(&[kx, ky, igw, igh].map(f32::to_bits));
        self.gpu_filter_dispatch(n, GPU_WARP, &[ks, kg], None, &[pass]);
        if log {
            eprintln!(
                "GPU warp node {}: shader {} gradient, {w}x{h} from {}x{} gradient {gw}x{gh}, k ({kx}, {ky}) tiling {wrap}",
                n.index, if same { "texel" } else { "bilinear" }, si.0, si.1
            );
        }
        true
    }

    /// The directional warp node (op 0x18) on the GPU (gpu_warp.wgsl), mirroring
    /// "directional_warp": the source is sampled at its own size, the intensity input at the
    /// node's size (input_at: an image.rs "resized" pass first when its size differs). The
    /// default intensity 10, the angle's cos/sin (f32, Rust's libm), the pixelsize and v9's third
    /// parameter (subtracted from the intensity input) give dx, dy and the offset here.
    /// CPU fallbacks: a missing input, the SDRT_WARPBIG / SDRT_TEXFLOAT / SDRT_DW* / resize
    /// switches, images over one storage binding or sides over 65536 texels, non-finite node
    /// constants (the node variables are restored).
    #[cfg(feature = "gpu")]
    fn try_gpu_dirwarp(&mut self, n: &Node) -> bool {
        use crate::gpu::Src;
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let early = |why: &str| {
            if log {
                eprintln!("GPU dirwarp node {}: CPU fallback ({why})", n.index);
            }
            false
        };
        if n.inputs.len() < 2 {
            return early("missing input");
        }
        let (ks, ka) = (n.inputs[0], n.inputs[1]);
        const SWITCHES: [&str; 8] = ["SDRT_WARPBIG", "SDRT_TEXFLOAT", "SDRT_DWDEF", "SDRT_DWDBG", "SDRT_DW3", "SDRT_RESIZE", "SDRT_RESIZEMIP", "SDRT_BIL256"];
        if SWITCHES.iter().any(|s| crate::envc::var(s).is_ok()) {
            return early("experiment switch");
        }
        let (Some(si), Some(ai)) = (self.gpu_image_info(ks), self.gpu_image_info(ka)) else { return early("missing input") };
        let (w, h) = n.size();
        let g = self.gpu.as_ref().unwrap();
        if !g.fits(si.0, si.1) || !g.fits(ai.0, ai.1) || !g.fits(w, h) {
            return early("image exceeds one GPU storage binding");
        }
        if [si.0, si.1, ai.0, ai.1].iter().any(|&s| s > 65536) {
            return early("input side over 65536 texels");
        }
        // Evaluate the seed and the parameters exactly as exec() and "directional_warp" do.
        let saved_vars = self.node_vars.clone();
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let inten = self.param(n, "intensity", seed).map_or(10.0, |v| v[0]);
        let ang = self.param(n, "angle", seed).map_or(0.0, |v| v[0]) * std::f32::consts::TAU;
        let ps = self.pixelsize(n, seed);
        let sub = self.param(n, "v9third", seed).map_or(0.0, |v| v[0]);
        let (dx, dy) = (ang.cos() * inten / 256.0 * ps.0, ang.sin() * inten / 256.0 * ps.1);
        if !(dx.is_finite() && dy.is_finite() && sub.is_finite()) {
            self.node_vars = saved_vars;
            return early("non-finite intensity, angle, pixelsize or offset");
        }
        let wrap = n.b3 & 3;
        let same = (ai.0, ai.1) == (w, h);
        let (fmt, afmt) = (self.tex_format(ks), self.tex_format(ka));
        let flags = wrap | u32::from(same) << 2 | u32::from(ai.2) << 3;
        let mut pass = Self::filter_pass(1, flags, [si.0 as u32, si.1 as u32, fmt, u32::from(si.2)], (w, h), n.color, n.format(), Src::Node(ks), Src::Node(ka));
        pass.config[4..8].copy_from_slice(&[ai.0 as u32, ai.1 as u32, afmt, u32::from(ai.2)]);
        pass.config[20..24].copy_from_slice(&[dx, dy, sub, 0.0].map(f32::to_bits));
        self.gpu_filter_dispatch(n, GPU_WARP, &[ks, ka], None, &[pass]);
        if log {
            eprintln!(
                "GPU dirwarp node {}: shader direct 1 passes, {w}x{h} from {}x{} intensity {}x{}, d ({dx}, {dy}) offset {sub} tiling {wrap}",
                n.index, si.0, si.1, ai.0, ai.1
            );
        }
        true
    }

    /// The distance node (op 0x2A) on the GPU (gpu_distance.wgsl), mirroring "distance" in its
    /// 14.1 mode (SDRT_DISTQ / SDRT_DIST14 unset): jump flooding, one GPU pass per step
    /// (max(w, h) / 2 down to 1) in the CPU's candidate order and with its strict-improvement tie
    /// rule, then a shading pass (the Euclidean root is correctly rounded, the falloff fused as
    /// the CPU's). The mask and source are input_at: an image.rs "resized" pass first when their
    /// size differs. The constants (1/pixelsize, distance / 256 and its reciprocal) are computed
    /// here as the CPU does. CPU fallbacks: a missing input, the SDRT_DIST* / resize switches,
    /// images over one storage binding, sides over 16384 texels, and (after the parameters were
    /// evaluated: the node variables are restored) a pixel size or maximum distance whose
    /// reciprocal is non-finite or outside the range the shader reproduces exactly.
    #[cfg(feature = "gpu")]
    fn try_gpu_distance(&mut self, n: &Node) -> bool {
        use crate::gpu::Src;
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let early = |why: &str| {
            if log {
                eprintln!("GPU distance node {}: CPU fallback ({why})", n.index);
            }
            false
        };
        let Some(&km) = n.inputs.first() else { return early("missing input") };
        const SWITCHES: [&str; 6] = ["SDRT_DISTQ", "SDRT_DIST14", "SDRT_DISTDBG", "SDRT_RESIZE", "SDRT_RESIZEMIP", "SDRT_BIL256"];
        if SWITCHES.iter().any(|s| crate::envc::var(s).is_ok()) {
            return early("experiment switch");
        }
        let ks = n.inputs.get(1).copied();
        let Some(mi) = self.gpu_image_info(km) else { return early("missing input") };
        let si = match ks {
            Some(k) => match self.gpu_image_info(k) {
                Some(i) => Some(i),
                None => return early("missing input"),
            },
            None => None,
        };
        let (w, h) = n.size();
        let g = self.gpu.as_ref().unwrap();
        if !g.fits(w, h) || !g.fits(mi.0, mi.1) || si.is_some_and(|i| !g.fits(i.0, i.1)) {
            return early("image exceeds one GPU storage binding");
        }
        if [w, h, mi.0, mi.1, si.map_or(0, |i| i.0), si.map_or(0, |i| i.1)].iter().any(|&s| s > 16384) {
            return early("side over 16384 texels");
        }
        // Evaluate the seed and the parameters exactly as exec() and "distance" do (the
        // "distance" function is evaluated twice there, with the pixel size and the mode between).
        let saved_vars = self.node_vars.clone();
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let _dist = self.param(n, "distance", seed).map_or(10.0, |v| v[0]) * w as f32 / 256.0;
        let ps = self.pixelsize(n, seed);
        let mode = self.param(n, "mode", seed).map_or((n.flags >> 4) & 3, |v| v[0] as u32);
        let dmax = self.param(n, "distance", seed).map_or(10.0, |v| v[0]) / 256.0;
        let inv = if dmax != 0.0 { 1.0f32 / dmax } else { 0.0f32 };
        let (ix, iy) = (1.0f32 / ps.0, 1.0f32 / ps.1);
        let scale_ok = |v: f32| v.is_finite() && (1e-6..=1e6).contains(&v.abs());
        if !(scale_ok(ix) && scale_ok(iy)) {
            self.node_vars = saved_vars;
            return early("pixel size outside the exact range");
        }
        if !(dmax.is_finite() && (dmax == 0.0 || (inv.is_finite() && inv.abs() < 1e20))) {
            self.node_vars = saved_vars;
            return early("maximum distance outside the exact range");
        }
        let mut passes = Vec::new();
        // the mask and the source at the node's size (resized: unquantized, own colour flag)
        let at_size = |passes: &mut Vec<crate::gpu::Pass>, k: usize, i: (usize, usize, bool)| {
            if (i.0, i.1) == (w, h) {
                Src::Node(k)
            } else {
                passes.push(Self::filter_pass(2, 0, [i.0 as u32, i.1 as u32, 3, u32::from(i.2)], (w, h), i.2, 3, Src::Node(k), Src::None));
                Src::Pass(passes.len() - 1)
            }
        };
        let mask = at_size(&mut passes, km, mi);
        let src = match (ks, si) {
            (Some(k), Some(i)) => Some(at_size(&mut passes, k, i)),
            _ => None,
        };
        let mut flags = (n.b3 & 3) | u32::from(mi.2) << 3 | mode.min(2) << 4;
        let mut cur = mask;
        let mut first = true;
        let mut step = w.max(h) / 2;
        let mut floods = 0;
        while step > 0 {
            let mut p = Self::filter_pass(0, flags | if first { 4 } else { 0 }, [0; 4], (w, h), true, 3, cur, Src::None);
            p.config[12..14].copy_from_slice(&[step as u32, w.max(h) as u32]);
            p.config[4..8].copy_from_slice(&[ix, iy, dmax, inv].map(f32::to_bits));
            passes.push(p);
            cur = Src::Pass(passes.len() - 1);
            first = false;
            step /= 2;
            floods += 1;
        }
        if src.is_some() {
            flags |= 64 | (n.flags & 2) << 6;
        }
        flags |= u32::from(n.color) << 9;
        let mut p = Self::filter_pass(1, flags | if first { 4 } else { 0 }, [0; 4], (w, h), n.color, n.format(), cur, src.unwrap_or(Src::None));
        p.config[4..8].copy_from_slice(&[ix, iy, dmax, inv].map(f32::to_bits));
        passes.push(p);
        let inputs: Vec<usize> = std::iter::once(km).chain(ks).collect();
        self.gpu_filter_dispatch(n, GPU_DISTANCE, &inputs, None, &passes);
        if log {
            eprintln!(
                "GPU distance node {}: shader {} passes ({floods} flood), {w}x{h}, mask {}x{} source {:?}, metric {}, tiling {}, dmax {dmax} 1/ps ({ix}, {iy})",
                n.index, passes.len(), mi.0, mi.1, si.map(|i| (i.0, i.1)), mode.min(2), n.b3 & 3
            );
        }
        true
    }

    /// The blur node (op 0x14) on the GPU (gpu_blur.wgsl), mirroring "blur": the taps of
    /// "blur_taps" / "blur_fetches" (per output parity) are computed here and bound as a table.
    /// - a larger input (both sides >= the node's, not equal): a horizontal pass of "bilinear_or"
    ///   fetches at its own size (offsets off / w), then "box_pass_parity" along y;
    /// - otherwise input_at (an image.rs "resized" pass when the size differs), then
    ///   "box_pass_parity" along x and along y.
    /// The output keeps the input's colour flag, as box_pass_parity does. CPU fallbacks: a missing
    /// input, the SDRT_BLUR* / SDRT_TEXFLOAT / resize switches, images over one storage binding or
    /// sides over 65536 texels, non-finite radii or radii over 4096 pixels (after the parameters
    /// were evaluated: the node variables are restored).
    #[cfg(feature = "gpu")]
    fn try_gpu_blur(&mut self, n: &Node) -> bool {
        use crate::gpu::Src;
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let early = |why: &str| {
            if log {
                eprintln!("GPU blur node {}: CPU fallback ({why})", n.index);
            }
            false
        };
        let Some(&k) = n.inputs.first() else { return early("missing input") };
        const SWITCHES: [&str; 7] = ["SDRT_BLURBIG", "SDRT_BLURYX", "SDRT_BLURBOX", "SDRT_TEXFLOAT", "SDRT_RESIZE", "SDRT_RESIZEMIP", "SDRT_BIL256"];
        if SWITCHES.iter().any(|s| crate::envc::var(s).is_ok()) {
            return early("experiment switch");
        }
        let Some((nw, nh, color)) = self.gpu_image_info(k) else { return early("missing input") };
        let (w, h) = n.size();
        let g = self.gpu.as_ref().unwrap();
        if !g.fits(nw, nh) || !g.fits(w, h) {
            return early("image exceeds one GPU storage binding");
        }
        if nw.max(nh) > 65536 {
            return early("input side over 65536 texels");
        }
        // Evaluate the seed and the parameters exactly as exec() and "blur" do.
        let saved_vars = self.node_vars.clone();
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let inten = self.param(n, "intensity", seed).map_or(10.0, |v| v[0]);
        let ps = self.pixelsize(n, seed);
        let rx = inten * w as f32 / 256.0 * ps.0;
        let ry = inten * h as f32 / 256.0 * ps.1;
        if !(rx.abs() <= 4096.0 && ry.abs() <= 4096.0) {
            self.node_vars = saved_vars;
            return early("non-finite or huge radius");
        }
        let wrap = n.b3 & 3;
        let big = (nw, nh) != (w, h) && nw >= w && nh >= h;
        let mut table: Vec<[f32; 4]> = Vec::new();
        // (start, count) of a tap list in the table
        let mut push = |taps: Vec<(f32, f32)>| -> [u32; 2] {
            let start = table.len() as u32;
            table.extend(taps.iter().map(|&(o, wt)| [o, wt, 0.0, 0.0]));
            [start, taps.len() as u32]
        };
        let box_taps = |r: f32, parity: i64| blur_taps(r, parity).into_iter().map(|(t, wt)| (t as f32, wt)).collect::<Vec<_>>();
        let node_desc = [w as u32, h as u32, 3, u32::from(color)];
        let mut passes = Vec::new();
        let first;
        if big {
            // u + off / w as f32: the offsets in u steps
            let steps = |r: f32, parity: i64| blur_fetches(r, parity).into_iter().map(|(off, wt)| (off / w as f32, wt)).collect::<Vec<_>>();
            let (e, o) = (push(steps(rx, 0)), push(steps(rx, 1)));
            let mut p = Self::filter_pass(1, wrap, [nw as u32, nh as u32, self.tex_format(k), u32::from(color)], (w, h), color, 3, Src::Node(k), Src::Node(GPU_TABLE));
            p.config[4..8].copy_from_slice(&[e[0], e[1], o[0], o[1]]);
            passes.push(p);
            first = Src::Pass(0);
        } else {
            let mut src = Src::Node(k);
            if (nw, nh) != (w, h) {
                passes.push(Self::filter_pass(2, 0, [nw as u32, nh as u32, 3, u32::from(color)], (w, h), color, 3, Src::Node(k), Src::None));
                src = Src::Pass(0);
            }
            let (e, o) = (push(box_taps(rx, 0)), push(box_taps(rx, 1)));
            let mut p = Self::filter_pass(0, wrap, node_desc, (w, h), color, 3, src, Src::Node(GPU_TABLE));
            p.config[4..8].copy_from_slice(&[e[0], e[1], o[0], o[1]]);
            passes.push(p);
            first = Src::Pass(passes.len() - 1);
        }
        let (e, o) = (push(box_taps(ry, 0)), push(box_taps(ry, 1)));
        let mut p = Self::filter_pass(0, wrap | 0x100, node_desc, (w, h), color, n.format(), first, Src::Node(GPU_TABLE));
        p.config[4..8].copy_from_slice(&[e[0], e[1], o[0], o[1]]);
        passes.push(p);
        if !self.gpu.as_ref().unwrap().fits(table.len(), 1) {
            self.node_vars = saved_vars;
            return early("tap table exceeds one GPU storage binding");
        }
        let taps = table.len();
        self.gpu_filter_dispatch(n, GPU_BLUR, &[k], Some(table), &passes);
        if log {
            eprintln!(
                "GPU blur node {}: shader {} {} passes, {w}x{h} from {nw}x{nh}, r ({rx}, {ry}), {taps} taps, tiling {wrap}",
                n.index, if big { "fetches" } else { "box" }, passes.len()
            );
        }
        true
    }

    /// The directional motion blur node (op 0x16) on the GPU (gpu_mblur.wgsl), mirroring
    /// "motion_blur" on input_at (an image.rs "resized" pass first when the size differs):
    /// - no direction (major == 0): the input itself;
    /// - one to three taps, or an axis-aligned direction: the point path, "bilinear" taps at the
    ///   offsets t·dx/major, t·dy/major (computed here);
    /// - otherwise the sheared image continued past the seam and blurred along the major axis
    ///   (padded by ep on an untiled minor axis), then sampled back at the unsheared positions;
    ///   the shear of every row (column) is computed here, the divisions by the padded size use a
    ///   correctly rounded GPU division (s_div).
    /// CPU fallbacks: a missing input, the SDRT_DMB* / SDRT_TEXFLOAT / resize switches, images over
    /// one storage binding or sides over 65536 texels, non-finite radii or radii over 4096 pixels
    /// (the node variables are restored).
    #[cfg(feature = "gpu")]
    fn try_gpu_motion_blur(&mut self, n: &Node) -> bool {
        use crate::gpu::Src;
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let early = |why: &str| {
            if log {
                eprintln!("GPU motionblur node {}: CPU fallback ({why})", n.index);
            }
            false
        };
        let Some(&k) = n.inputs.first() else { return early("missing input") };
        const SWITCHES: [&str; 10] = ["SDRT_TEXFLOAT", "SDRT_BIL256", "SDRT_DMB", "SDRT_DMBSEAM", "SDRT_DMBPAD", "SDRT_DMBQ", "SDRT_DMBEXT", "SDRT_RESIZE", "SDRT_RESIZEMIP", "SDRT_DMBRADIUS"];
        if SWITCHES.iter().any(|s| crate::envc::var(s).is_ok()) {
            return early("experiment switch");
        }
        let Some((nw, nh, scolor)) = self.gpu_image_info(k) else { return early("missing input") };
        let (w, h) = n.size();
        let g = self.gpu.as_ref().unwrap();
        if !g.fits(nw, nh) || !g.fits(w, h) {
            return early("image exceeds one GPU storage binding");
        }
        if nw.max(nh) > 65536 {
            return early("input side over 65536 texels");
        }
        // Evaluate the seed and the parameters exactly as exec() and "motion_blur" do.
        let saved_vars = self.node_vars.clone();
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let inten = self.param(n, "intensity", seed).map_or(10.0, |v| v[0]);
        let ang = self.param(n, "angle", seed).map_or(0.0, |v| v[0]) * std::f32::consts::TAU;
        let wrap = n.b3 & 3;
        let ps = self.pixelsize(n, seed);
        let r = inten * w as f32 / 256.0;
        let (dx, dy) = (ang.cos() * ps.0, ang.sin() * ps.1);
        if crate::envc::var("SDRT_STRICT_TRACE").is_ok() {
            eprintln!("STRICT {{\"k\":{},\"op\":22,\"input\":{k},\"sfmt\":{},\"fmt\":{},\"sw\":{nw},\"sh\":{nh},\"w\":{w},\"h\":{h},\"flags\":{},\"tiling\":{wrap},\"intensity\":{inten},\"angle_radians\":{ang},\"dx\":{dx},\"dy\":{dy},\"r\":{r}}}", n.index, self.p.nodes[k].format(), n.format(), n.flags);
        }
        let y_major = dy.abs() >= dx.abs();
        let major = if y_major { dy.abs() } else { dx.abs() };
        let fmt = self.tex_format(k);
        let mut passes = Vec::new();
        // input_at: resized to the node (unquantized, its colour flag)
        let src = if (nw, nh) == (w, h) {
            Src::Node(k)
        } else {
            passes.push(Self::filter_pass(3, 0, [nw as u32, nh as u32, 3, u32::from(scolor)], (w, h), scolor, 3, Src::Node(k), Src::None));
            Src::Pass(0)
        };
        let path;
        let mut table: Vec<[f32; 4]> = Vec::new();
        if major == 0.0 {
            // the input itself, stored in the node's format with the input's colour flag
            if let Some(p) = passes.first_mut() {
                p.format = n.format();
            } else {
                passes.push(Self::filter_pass(4, 0, [w as u32, h as u32, 3, u32::from(scolor)], (w, h), scolor, n.format(), src, Src::None));
            }
            path = "copy";
        } else {
            let axis_aligned = dx.abs().min(dy.abs()) < 1e-6;
            let radius = if axis_aligned { r * major } else { 0.5 + (r - 0.5) * major };
            if !(radius.abs() <= 4096.0) {
                self.node_vars = saved_vars;
                return early("non-finite or huge radius");
            }
            let taps = box_taps(radius);
            let desc = [w as u32, h as u32, fmt, u32::from(scolor)];
            if taps.len() <= 3 || axis_aligned {
                for &(t, wt) in &taps {
                    let t = t as f32;
                    table.push([t * dx / major, t * dy / major, wt, 0.0]);
                }
                let mut p = Self::filter_pass(0, wrap, desc, (w, h), n.color, n.format(), src, Src::Node(GPU_TABLE));
                p.config[4..8].copy_from_slice(&[0, taps.len() as u32, 0, 0]);
                passes.push(p);
                path = "point";
            } else {
                // an untiled minor axis: the sheared image is wider by the largest shear
                let ext = if y_major { wrap & 1 == 0 } else { wrap & 2 == 0 };
                let ep = if ext {
                    let smax = if y_major { h as f32 * 0.5 * (dx / dy).abs() } else { w as f32 * 0.5 * (dy / dx).abs() };
                    smax.ceil() as usize + 2
                } else {
                    0
                };
                let (bw, bh) = if y_major { (w + 2 * ep, h) } else { (w, h + 2 * ep) };
                if !self.gpu.as_ref().unwrap().fits(bw, bh) {
                    self.node_vars = saved_vars;
                    return early("padded image exceeds one GPU storage binding");
                }
                let kmax = taps.iter().map(|&(t, _)| t.abs()).max().unwrap_or(0);
                table.extend(taps.iter().map(|&(t, wt)| [t as f32, wt, 0.0, 0.0]));
                // the shear of major-axis index j, as the CPU computes it for a row (column)
                let along = if y_major { h } else { w } as i64;
                for j in -kmax..along + kmax {
                    let j = j as f32;
                    let s = if y_major { (j + 0.5 - h as f32 * 0.5) * dx / dy } else { (j + 0.5 - w as f32 * 0.5) * dy / dx };
                    table.push([s, 0.0, 0.0, 0.0]);
                }
                let values = [0, taps.len() as u32, taps.len() as u32, kmax as u32];
                let flags = wrap | u32::from(y_major) << 2;
                let mut shear = Self::filter_pass(1, flags, desc, (bw, bh), n.color, 3, src, Src::Node(GPU_TABLE));
                shear.config[4..8].copy_from_slice(&values);
                shear.config[8] = ep as u32;
                passes.push(shear);
                let mut unshear = Self::filter_pass(2, flags, [bw as u32, bh as u32, fmt, u32::from(n.color)], (w, h), n.color, n.format(), Src::Pass(passes.len() - 1), Src::Node(GPU_TABLE));
                unshear.config[4..8].copy_from_slice(&values);
                unshear.config[8] = ep as u32;
                passes.push(unshear);
                path = if ep > 0 { "shear-padded" } else { "shear" };
            }
        }
        if table.iter().flatten().any(|v| !v.is_finite()) || !self.gpu.as_ref().unwrap().fits(table.len().max(1), 1) {
            self.node_vars = saved_vars;
            return early("non-finite or oversized tap table");
        }
        let entries = table.len();
        self.gpu_filter_dispatch(n, GPU_MBLUR, &[k], (!table.is_empty()).then_some(table), &passes);
        if log {
            eprintln!(
                "GPU motionblur node {}: shader {path} {} passes, {w}x{h} from {nw}x{nh}, d ({dx}, {dy}) r {r}, {entries} table entries, tiling {wrap}",
                n.index, passes.len()
            );
        }
        true
    }

    /// Node k's size and colour flag: resident on the GPU, or computed on the CPU.
    #[cfg(feature = "gpu")]
    fn gpu_input(&self, k: usize) -> Option<(usize, usize, bool)> {
        let g = self.gpu.as_ref().unwrap();
        g.size(k)
            .map(|(w, h)| (w, h, g.color(k).unwrap()))
            .or_else(|| self.images.get(k).and_then(|i| i.as_ref()).map(|im| (im.w, im.h, im.color)))
    }

    /// One pass of a colour kernel (gradient map, curve, HSL, emboss, dynamic gradient) for node
    /// n: input_a / input_b are graph nodes (uploaded when the CPU computed them), or input_b is
    /// a table built on the CPU, uploaded for this dispatch only (as GPU_TABLE, released after
    /// the submission is recorded). The output becomes resident node n.
    #[cfg(feature = "gpu")]
    fn gpu_color_pass(&mut self, n: &Node, kernel: (&'static str, &'static str), config: [u32; 40], color: bool, a: Option<usize>, b: Option<usize>, table: Option<Image>) {
        use crate::gpu::{Pass, Src};
        let (w, h) = n.size();
        let g = self.gpu.as_mut().unwrap();
        for k in [a, b].into_iter().flatten() {
            if g.size(k).is_none() {
                g.upload(k, self.images[k].as_ref().expect("GPU kernel input computed")).expect("GPU upload failed");
            }
        }
        let src_b = match &table {
            Some(t) => {
                g.release(GPU_TABLE);
                g.upload(GPU_TABLE, t).expect("GPU table upload failed");
                Src::Node(GPU_TABLE)
            }
            None => b.map_or(Src::None, Src::Node),
        };
        let pass = Pass { config, w, h, color, format: n.format(), a: a.map_or(Src::None, Src::Node), b: src_b };
        g.dispatch_passes(n.index, kernel, &[pass]).unwrap_or_else(|e| panic!("Vulkan {} dispatch failed: {e}", kernel.0));
        if table.is_some() {
            g.release(GPU_TABLE);
        }
        self.images[n.index] = None;
    }

    /// The gradient map (op 0x00) on the GPU (gpu_gradient.wgsl). The table is built on the CPU
    /// exactly as "gradient_map" builds it (same keys, same rules per version and format):
    /// v8word or a float node format: the 1024-texel f32 table ("gradient_lut_f32"), sampled as
    /// "sample_f32_lut" (f64 mix reproduced in integer arithmetic) with half-float truncation;
    /// otherwise the 256-texel UNORM16 table ("gradient_lut16") interpolated in integers.
    /// CPU fallbacks: a missing input, an input of another size (the CPU resizes it), the
    /// SDRT_GM* switches, images over one storage binding, and f32 tables the integer mix does
    /// not cover (non-finite texels, magnitudes below 2^-95, neighbours 2^31 apart).
    #[cfg(feature = "gpu")]
    fn try_gpu_gradient_map(&mut self, n: &Node) -> bool {
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let early = |why: &str| {
            if log {
                eprintln!("GPU gradient map node {}: CPU fallback ({why})", n.index);
            }
            false
        };
        let k = n.inputs[0];
        if k >= self.p.nodes.len() {
            return early("missing input");
        }
        if ["SDRT_GMREP", "SDRT_GMDBG", "SDRT_GMV8MID", "SDRT_GMV8", "SDRT_GMLUT"].iter().any(|s| crate::envc::var(s).is_ok()) {
            return early("experiment switch");
        }
        let (w, h) = n.size();
        let Some((sw, sh, src_color)) = self.gpu_input(k) else { return early("input not computed") };
        if (sw, sh) != (w, h) {
            return early("input of another size");
        }
        if !self.gpu.as_ref().unwrap().fits(w, h) {
            return early("image exceeds one GPU storage binding");
        }
        // exactly as exec() + gradient_map: the seed (a function may set node variables), v8word
        let saved_vars = self.node_vars.clone();
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let v8w_param = self.param(n, "v8word", seed).and_then(|v| v.first().copied()).unwrap_or(0.0);
        let (version, b3) = (self.p.version, n.b3);
        let has_v8w = version >= 8 && b3 & 2 != 0;
        let repeat = if version >= 8 { b3 & 8 != 0 } else { b3 & 4 != 0 };
        let mid = b3 & 1 != 0;
        let smooth = if version >= 8 && mid { true } else { !mid && version < 8 };
        let v8w = if has_v8w { v8w_param } else { 0.0 };
        let keys = self.gradient_keys(n);
        let float_table = has_v8w || n.format() >= 2;
        let table: Vec<[f32; 4]> = if float_table {
            let lut = gradient_lut_f32(&keys, has_v8w, v8w, smooth);
            if let Err(why) = gpu_mixable(&lut, repeat) {
                self.node_vars = saved_vars;
                return early(&format!("f32 table outside the integer mix's range, {why}"));
            }
            lut.to_vec()
        } else {
            // 16-bit codes as exact floats
            gradient_lut16(&keys, v8w, smooth, 256).iter().map(|t| t.map(|c| c as f32)).collect()
        };
        let mut config = [0u32; 40];
        config[0] = u32::from(!float_table);
        config[3] = u32::from(repeat) | u32::from(float_table && n.format() == 2) << 1 | u32::from(src_color) << 2;
        let lut = Image { w: table.len(), h: 1, color: true, px: table };
        self.gpu_color_pass(n, GPU_GRADIENT, config, n.color, Some(k), None, Some(lut));
        if log {
            eprintln!(
                "GPU gradient map node {}: shader {} table, {} keys, repeat {repeat}, v{version} b3 {b3:#x}",
                n.index, if float_table { "f32" } else { "unorm16" }, keys.len()
            );
        }
        true
    }

    /// HSL (op 0x1C) on the GPU (gpu_hsl.wgsl): "hsl", "rgb_to_hsl", "hsl_to_rgb" per pixel.
    /// CPU fallbacks: a missing input, an input of another size, images over one binding.
    #[cfg(feature = "gpu")]
    fn try_gpu_hsl(&mut self, n: &Node) -> bool {
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let early = |why: &str| {
            if log {
                eprintln!("GPU hsl node {}: CPU fallback ({why})", n.index);
            }
            false
        };
        let k = n.inputs[0];
        if k >= self.p.nodes.len() {
            return early("missing input");
        }
        let (w, h) = n.size();
        let Some((sw, sh, _)) = self.gpu_input(k) else { return early("input not computed") };
        if (sw, sh) != (w, h) {
            return early("input of another size");
        }
        if !self.gpu.as_ref().unwrap().fits(w, h) {
            return early("image exceeds one GPU storage binding");
        }
        // exactly as exec() + hsl: the seed, then hue, saturation, luminosity
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let hue = self.param(n, "hue", seed).map_or(0.5, |v| v[0]);
        let sat = self.param(n, "saturation", seed).map_or(0.5, |v| v[0]);
        let lum = self.param(n, "luminosity", seed).map_or(0.5, |v| v[0]);
        let mut config = [0u32; 40];
        // the shifts as the CPU computes them for every pixel, and its f32 constants
        config[4..8].copy_from_slice(&[(hue - 0.5) * 2.0, 2.0 * (sat - 0.5), 2.0 * (lum - 0.5), 0.0].map(f32::to_bits));
        config[8..12].copy_from_slice(&[1.0f32 / 3.0, 1.0 / 6.0, 2.0 / 3.0, 0.0].map(f32::to_bits));
        // the CPU's output image is colour whatever the node's flag
        self.gpu_color_pass(n, GPU_HSL, config, true, Some(k), None, None);
        if log {
            eprintln!("GPU hsl node {}: shader, hue {hue} saturation {sat} luminosity {lum}", n.index);
        }
        true
    }

    /// Emboss (op 0x10) on the GPU (gpu_emboss.wgsl): "emboss" per pixel; a height of the node's
    /// size takes texel differences, any other size three texture-unit samples (the CPU reads
    /// it at its own size). The light angle's cos/sin are computed once on the CPU.
    /// CPU fallbacks: missing inputs, a source of another size, SDRT_WARPBIG / SDRT_TEXFLOAT,
    /// images over one binding.
    #[cfg(feature = "gpu")]
    fn try_gpu_emboss(&mut self, n: &Node) -> bool {
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let early = |why: &str| {
            if log {
                eprintln!("GPU emboss node {}: CPU fallback ({why})", n.index);
            }
            false
        };
        let (a, hk) = (n.inputs[0], n.inputs[1]);
        if a >= self.p.nodes.len() || hk >= self.p.nodes.len() {
            return early("missing input");
        }
        if ["SDRT_WARPBIG", "SDRT_TEXFLOAT"].iter().any(|s| crate::envc::var(s).is_ok()) {
            return early("experiment switch");
        }
        let (w, h) = n.size();
        let (Some((sw, sh, _)), Some((hw, hh, hcolor))) = (self.gpu_input(a), self.gpu_input(hk)) else { return early("input not computed") };
        if (sw, sh) != (w, h) {
            return early("source of another size");
        }
        let g = self.gpu.as_ref().unwrap();
        if !g.fits(w, h) || !g.fits(hw, hh) {
            return early("image exceeds one GPU storage binding");
        }
        let hfmt = self.tex_format(hk);
        // exactly as exec() + emboss: the seed, intensity, light angle, highlight, shadow, pixel size
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let inten = self.param(n, "intensity", seed).map_or(5.0, |v| v[0]);
        let ang = self.param(n, "lightangle", seed).map_or(0.0, |v| v[0]) * std::f32::consts::TAU;
        let hl = self.param(n, "highlight", seed).unwrap_or(vec![1.0; 4]);
        let sh = self.param(n, "shadow", seed).unwrap_or(vec![0.0, 0.0, 0.0, 1.0]);
        let ps = self.pixelsize(n, seed);
        let same = hw == w && hh == h;
        let (kx, ky) = (inten * hw as f32 / 256.0 * ps.0, inten * hh as f32 / 256.0 * ps.1);
        let (ca, sa) = (ang.cos(), ang.sin());
        let ch = |c: usize| if n.color { c } else { 0 };
        let mut config = [0u32; 40];
        config[3] = (n.b3 & 3) | u32::from(same) << 2 | u32::from(hcolor) << 3;
        config[4..8].copy_from_slice(&[kx, ky, ca, sa].map(f32::to_bits));
        config[8..12].copy_from_slice(&[1.0 / hw as f32, 1.0 / hh as f32, 0.0, 0.0].map(f32::to_bits));
        config[12..16].copy_from_slice(&[hw as u32, hh as u32, hfmt, u32::from(hcolor)]);
        config[20..24].copy_from_slice(&[hl[ch(0)], hl[ch(1)], hl[ch(2)], 0.0].map(f32::to_bits));
        config[24..28].copy_from_slice(&[1.0 - sh[ch(0)], 1.0 - sh[ch(1)], 1.0 - sh[ch(2)], 0.0].map(f32::to_bits));
        self.gpu_color_pass(n, GPU_EMBOSS, config, n.color, Some(a), Some(hk), None);
        if log {
            eprintln!("GPU emboss node {}: shader, height {hw}x{hh} ({}), kx {kx} ky {ky} angle {ang}", n.index, if same { "texel differences" } else { "sampled" });
        }
        true
    }

    /// The dynamic gradient (op 0x26) on the GPU (gpu_dyngrad.wgsl): "dyn_gradient" per pixel,
    /// the gradient image sampled by the texture unit's bilinear at its own size.
    /// CPU fallbacks: missing inputs, a source of another size, SDRT_TEXFLOAT, a gradient with a
    /// side that is not a power of two (inexact reciprocal), a non-finite or huge coordinate,
    /// images over one binding.
    #[cfg(feature = "gpu")]
    fn try_gpu_dyn_gradient(&mut self, n: &Node) -> bool {
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let early = |why: &str| {
            if log {
                eprintln!("GPU dynamic gradient node {}: CPU fallback ({why})", n.index);
            }
            false
        };
        let (a, gk) = (n.inputs[0], n.inputs[1]);
        if a >= self.p.nodes.len() || gk >= self.p.nodes.len() {
            return early("missing input");
        }
        if crate::envc::var("SDRT_TEXFLOAT").is_ok() {
            return early("experiment switch");
        }
        let (w, h) = n.size();
        let (Some((sw, sh, src_color)), Some((gw, gh, gcolor))) = (self.gpu_input(a), self.gpu_input(gk)) else { return early("input not computed") };
        if (sw, sh) != (w, h) {
            return early("source of another size");
        }
        let g = self.gpu.as_ref().unwrap();
        if !g.fits(w, h) || !g.fits(gw, gh) {
            return early("image exceeds one GPU storage binding");
        }
        if !gw.is_power_of_two() || !gh.is_power_of_two() {
            return early("gradient side not a power of two");
        }
        let gfmt = self.tex_format(gk);
        // exactly as exec() + dyn_gradient: the seed, then the coordinate
        let saved_vars = self.node_vars.clone();
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let coord = self.param(n, "coordinate", seed).map_or(0.0, |v| v[0]);
        // the sampler's texel index is an i32 on the GPU (i64 on the CPU)
        if !coord.is_finite() || coord.abs() * gw.max(gh) as f32 >= 16_777_216.0 {
            self.node_vars = saved_vars;
            return early("coordinate out of range");
        }
        let (repeat, vertical) = (n.b3 & 1 != 0, n.b3 & 8 != 0);
        let mut config = [0u32; 40];
        config[3] = u32::from(repeat) | u32::from(vertical) << 1 | u32::from(src_color) << 2;
        config[4..8].copy_from_slice(&[coord, gw as f32 - 1.0, gh as f32 - 1.0, 1.0 / gw as f32].map(f32::to_bits));
        config[8] = (1.0 / gh as f32).to_bits();
        config[12..16].copy_from_slice(&[gw as u32, gh as u32, gfmt, u32::from(gcolor)]);
        self.gpu_color_pass(n, GPU_DYNGRAD, config, n.color, Some(a), Some(gk), None);
        if log {
            eprintln!("GPU dynamic gradient node {}: shader, gradient {gw}x{gh} format {gfmt}, coordinate {coord}, repeat {repeat}, vertical {vertical}", n.index);
        }
        true
    }

    /// The curve (op 0x2C) on the GPU (gpu_curve.wgsl). The CPU evaluates each curve through a
    /// table of 1025 "bezier_curve" values interpolated with 1/256 fractions; that very table
    /// is built here and uploaded, the kernel does the per-pixel lookup.
    /// CPU fallbacks: a missing input, an input of another size (unless the curve runs along x
    /// and does not read it), the SDRT_CURVE* switches, tables outside the normal f32 range
    /// (the GPU flushes subnormal results), images over one binding.
    #[cfg(feature = "gpu")]
    fn try_gpu_curve(&mut self, n: &Node) -> bool {
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let early = |why: &str| {
            if log {
                eprintln!("GPU curve node {}: CPU fallback ({why})", n.index);
            }
            false
        };
        let k = n.inputs[0];
        if k >= self.p.nodes.len() {
            return early("missing input");
        }
        if ["SDRT_CURVEX", "SDRT_CURVEXT", "SDRT_CURVELUT", "SDRT_CURVELUTN", "SDRT_CURVELUTQ"].iter().any(|s| crate::envc::var(s).is_ok()) {
            return early("experiment switch");
        }
        let (w, h) = n.size();
        let Some((sw, sh, _)) = self.gpu_input(k) else { return early("input not computed") };
        let along_x = n.b3 & 2 != 0;
        if (sw, sh) != (w, h) && !along_x {
            return early("input of another size");
        }
        if !self.gpu.as_ref().unwrap().fits(w, h) {
            return early("image exceeds one GPU storage binding");
        }
        // exec() evaluates the seed (curve takes no parameter)
        let saved_vars = self.node_vars.clone();
        self.cur_size = [w as f32, h as f32];
        self.seed(n);
        let curves = self.curve_points(n);
        let tables: Vec<Vec<f32>> = curves.iter().map(|c| (0..=1024).map(|i| bezier_curve(c, i as f32 / 1024.0)).collect()).collect();
        let normal = |v: f32| v == 0.0 || (v.is_finite() && v.abs() >= f32::from_bits(37 << 23) && v.abs() <= f32::from_bits(253 << 23));
        if !tables.iter().zip(&curves).all(|(t, c)| c.is_empty() || t.iter().all(|&v| normal(v))) {
            self.node_vars = saved_vars;
            return early("table outside the normal f32 range");
        }
        let flat: Vec<f32> = tables.concat();
        let px: Vec<[f32; 4]> = flat.chunks(4).map(|c| std::array::from_fn(|j| c.get(j).copied().unwrap_or(0.0))).collect();
        let mut config = [0u32; 40];
        config[3] = u32::from(along_x) | u32::from(n.color) << 1;
        for (c, pts) in curves.iter().enumerate() {
            if pts.is_empty() {
                config[3] |= 1 << (8 + c);
            }
        }
        let table = Image { w: px.len(), h: 1, color: true, px };
        self.gpu_color_pass(n, GPU_CURVE, config, n.color, if along_x { None } else { Some(k) }, None, Some(table));
        if log {
            eprintln!("GPU curve node {}: shader, {} curves ({:?} points){}", n.index, curves.len(), curves.iter().map(Vec::len).collect::<Vec<_>>(), if along_x { ", along x" } else { "" });
        }
        true
    }

    #[cfg(feature = "gpu")]
    fn materialize_for_node(&mut self, node: &Node) {
        // Every image input and image parameter of a CPU node is read back. A selective version
        // (only the inputs each op reads) missed inputs that some ops read conditionally (SVG and
        // text backgrounds, dynamic PP sampler slots, input_at in helpers) and panicked with
        // "input computed" on real files; readbacks are cheap next to a CPU node.
        for &k in &node.inputs { self.materialize(k); }
        for (_, slot) in &node.params {
            if let Slot::Input(k) = slot { self.materialize(*k); }
        }
    }

    #[cfg(feature = "gpu")]
    fn materialize(&mut self, k: usize) {
        if k >= self.images.len() || self.images[k].is_some() {
            return;
        }
        if let Some(g) = &mut self.gpu {
            if g.size(k).is_some() {
                self.images[k] = Some(std::sync::Arc::new(g.download(k).expect("GPU readback failed")));
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn try_gpu_fxmap(&mut self, n: &Node) -> bool {
        use crate::gpu::{Pass, Src};
        let log = crate::envc::var("SDRT_GPU_LOG").is_ok();
        let fallback = |why: &str| {
            if log { eprintln!("GPU fxmap node {}: CPU fallback ({why})", n.index); }
            false
        };
        let switches = ["SDRT_COVER", "SDRT_LX64", "SDRT_SNAP", "SDRT_SBSROT", "SDRT_NEARMIP",
            "SDRT_SWPARENT", "SDRT_CNTPARENT", "SDRT_OUT0LATE", "SDRT_ITERREV", "SDRT_CORDER", "SDRT_PIXDBG", "SDRT_LIBMTRIG"];
        if std::env::vars_os().any(|(k,_)| {
            let k=k.to_string_lossy();
            (k.starts_with("SDRT_FX") && k!="SDRT_FXPROF" && k!="SDRT_FXPARDBG") || switches.iter().any(|s|k.starts_with(s))
        }) { return fallback("FX-Map experiment switch"); }
        let (w,h)=n.size();
        if !self.gpu.as_ref().unwrap().fits(w,h) { return fallback("output exceeds GPU binding"); }
        for &k in &n.inputs { self.materialize(k); }
        for (_,slot) in &n.params { if let Slot::Input(k)=slot { self.materialize(*k); } }
        let vars=self.node_vars.clone();
        self.cur_size=[w as f32,h as f32];
        let seed=self.seed(n);
        let plan=match crate::fxmap::prepare_gpu(self,n,seed) {
            Ok(p)=>p,
            Err(why)=> {self.node_vars=vars; return fallback(&why);}
        };
        let g=self.gpu.as_mut().unwrap();
        // Two temporary uploads share the existing image-buffer mechanism. The background
        // retains the CPU's resize and initial quantization semantics.
        const BG:usize=usize::MAX-1;
        g.release(BG); g.release(GPU_TABLE);
        g.upload(BG,&plan.background).expect("FX-Map background upload failed");
        let chunks=plan.tables.len();
        for (i,table) in plan.tables.iter().enumerate() {
            g.release(GPU_TABLE);
            g.upload(GPU_TABLE,table).expect("FX-Map table upload failed");
            let mut config=[0u32;40]; config[12]=plan.tiles_x; config[13]=n.format();
            if let Some(roi)=plan.roi {config[0]=1;for (j,v) in roi.iter().enumerate() {config[20+j]=v.to_bits();}}
            // Per-layer UNORM8 still uses the original format. Every intermediate
            // result is stored as f32; half truncation/final quantization run once.
            let pass=Pass {config,w,h,color:n.color,format:if i+1==chunks {n.format()} else {3},
                a:Src::Node(if i==0 {BG} else {n.index}),b:Src::Node(GPU_TABLE)};
            g.dispatch_passes(n.index,("fxmap",concat!(include_str!("gpu_color_common.wgsl"),include_str!("gpu_fxmap.wgsl"))),&[pass]).expect("Vulkan FX-Map dispatch failed");
        }
        g.release(BG); g.release(GPU_TABLE);
        self.images[n.index]=None;
        if log {eprintln!("GPU fxmap node {}: shader ordered-tiles chunks={chunks}",n.index);}
        true
    }

    #[cfg(feature = "gpu")]
    fn try_gpu(&mut self, n: &Node) -> bool {
        if self.gpu.is_none() {
            return false;
        }
        // research aid: SDRT_GPU_OPS=00,2c,... (hex opcodes) keeps every other node on the CPU,
        // so a real-file check of these kernels sees CPU inputs (identical in both backends)
        if let Ok(ops) = crate::envc::var("SDRT_GPU_OPS") {
            if !ops.split(',').any(|o| u32::from_str_radix(o.trim(), 16) == Ok(n.op)) {
                return false;
            }
        }
        if (n.op==0x02 && ["SDRT_MIX","SDRT_DIVNAN"].iter().any(|s|crate::envc::var(s).is_ok()))
            || (n.op==0x1e && crate::envc::var("SDRT_LVPCLAMP").as_deref()==Ok("0")) {
            if crate::envc::var("SDRT_GPU_LOG").is_ok() {
                let kern = if n.op == 0x02 { "blend" } else { "levels" };
                eprintln!("GPU {kern} node {}: CPU fallback (switch disabled)", n.index);
            }
            return false;
        }
        if n.op == 0x28 && n.flags >> 16 == 0 {
            // Inputs that cannot be bound (over one storage binding) keep the CPU path; this is
            // checked before any seed/parameter side effects.
            let too_big = n.inputs.iter().filter(|&&k| k < self.p.nodes.len()).any(|&k| {
                let g = self.gpu.as_ref().unwrap();
                g.size(k).or_else(|| self.images[k].as_ref().map(|im| (im.w, im.h))).is_some_and(|(w, h)| !g.fits(w, h))
            });
            if too_big {
                if crate::envc::var("SDRT_GPU_LOG").is_ok() {
                    eprintln!("GPU PP node {}: CPU fallback (input exceeds one GPU storage binding)", n.index);
                }
                return false;
            }
            if n.inputs.len() > 2 {
                let g = self.gpu.as_ref().unwrap();
                let mut total = 0u64;
                for &k in &n.inputs {
                    if k < self.p.nodes.len() {
                        if let Some((sw, sh)) = g.size(k).or_else(|| self.images.get(k).and_then(|i| i.as_ref()).map(|im| (im.w, im.h))) {
                            total += (sw * sh * 16) as u64;
                        }
                    }
                }
                if total.max(16) > g.limits().max_storage_buffer_binding_size as u64 {
                    if crate::envc::var("SDRT_GPU_LOG").is_ok() {
                        eprintln!("GPU PP node {}: CPU fallback (packed inputs exceed GPU binding size limit)", n.index);
                    }
                    return false;
                }
            }
            let f = self.func(n.data);
            crate::gpu_pp::NODE.with(|c| c.set(n.index));
            let mut shader = match crate::gpu_pp::compile(&f, &|uid| self.input(uid), &|k| self.vp_slot(k), &self.node_vars) {
                Ok(s) => s,
                Err(reason) => {
                    if crate::envc::var("SDRT_GPU_LOG").is_ok() {
                        eprintln!("GPU PP node {}: CPU fallback ({reason})", n.index);
                    }
                    return false;
                }
            };
            let (w, h) = n.size();
            self.cur_size = [w as f32, h as f32];
            let seed = self.seed(n);
            // Seed functions may update variables initialized by the size function. The
            // preflight above establishes support before seed side effects; only literals change.
            if f.ins.iter().any(|i| i.op == 4) {
                shader = crate::gpu_pp::compile(&f, &|uid| self.input(uid), &|k| self.vp_slot(k), &self.node_vars).expect("PP preflight already passed");
            }
            let mut config = [0u32; 40];
            config[3] = seed;
            let mut ids = [None; 2];
            let mut sources=Vec::new();
            for (slot, &k) in n.inputs.iter().enumerate() {
                if k >= self.p.nodes.len() {
                    sources.push((None,3)); continue;
                }
                sources.push((Some(k),self.tex_format(k)));
                let size = self.gpu.as_ref().unwrap().size(k).or_else(|| self.images[k].as_ref().map(|im| (im.w, im.h))).expect("PP input computed");
                if slot<2 {config[4 + slot * 4..8 + slot * 4].copy_from_slice(&[size.0 as u32, size.1 as u32, self.tex_format(k), u32::from(self.p.nodes[k].color)]);}
                let g = self.gpu.as_mut().unwrap();
                if g.size(k).is_none() {
                    g.upload(k, self.images[k].as_ref().expect("PP CPU input computed")).expect("PP upload failed");
                }
                if slot<2 {ids[slot] = Some(k);}
            }
            if crate::envc::var("SDRT_GPU_LOG").is_ok() {
                // logged before the dispatch: pipeline creation (the driver's compiler) is the step
                // that can take long for big control-flow shaders
                eprintln!("GPU PP node {}: shader ({} instructions, {} WGSL bytes, {} values{})", n.index, f.ins.len(), shader.text.len(), shader.lits.len(), if shader.flow { ", control flow" } else { "" });
            }
            if let Ok(dir) = crate::envc::var("SDRT_GPU_DUMP") {
                // research aid: the generated WGSL of each PP node, DIR/pp_<node>.wgsl, its
                // per-node values and the function's instructions
                let dir = std::path::Path::new(&dir);
                let _ = std::fs::create_dir_all(dir);
                let _ = std::fs::write(dir.join(format!("pp_{}.wgsl", n.index)), &shader.text);
                let lits: Vec<String> = shader.lits.iter().zip(&shader.moved).map(|(w, id)| format!("{w:#010x} {id}")).collect();
                let _ = std::fs::write(dir.join(format!("pp_{}.lits", n.index)), lits.join("\n") + "\n");
                let ins: Vec<String> = f.ins.iter().enumerate().map(|(k, i)| format!("{k}: op {} ty {} {:?}", i.op, i.ty, &*i.ops)).collect();
                let _ = std::fs::write(dir.join(format!("pp_{}.ins", n.index)), ins.join("\n") + "\n");
            }
            self.gpu.as_mut().unwrap().dispatch(n.index, w, h, n.color, n.format(), &config, ids[0], ids[1], Some((&shader.text, &shader.lits)), &sources).expect("Vulkan PP dispatch failed");
            self.images[n.index] = None;
            return true;
        }
        if n.op == 0x04 {
            // try_gpu_transform logs its own fallback reasons.
            return self.try_gpu_transform(n);
        }
        if n.op == 0x08 { return self.try_gpu_fxmap(n); }
        // The filter ports log their own fallback reasons too.
        if n.op == 0x0E { return self.try_gpu_warp(n); }
        if n.op == 0x18 { return self.try_gpu_dirwarp(n); }
        if n.op == 0x14 { return self.try_gpu_blur(n); }
        if n.op == 0x2A { return self.try_gpu_distance(n); }
        if n.op == 0x16 { return self.try_gpu_motion_blur(n); }
        // the colour kernels log their own fallback reasons too
        if n.op == 0x00 { return self.try_gpu_gradient_map(n); }
        if n.op == 0x1C { return self.try_gpu_hsl(n); }
        if n.op == 0x10 { return self.try_gpu_emboss(n); }
        if n.op == 0x26 { return self.try_gpu_dyn_gradient(n); }
        if n.op == 0x2C { return self.try_gpu_curve(n); }
        if !matches!(n.op, 0x0c | 0x06 | 0x02 | 0x1e | 0x24 | 0x1a) {
            return false;
        }
        let kern = match n.op {
            0x02 => "blend",
            0x1e => "levels",
            0x24 => "normal",
            0x1a => "sharpen",
            _ => "atomic",
        };
        let (w, h) = n.size();
        let g = self.gpu.as_ref().unwrap();
        if !g.fits(w, h) {
            return false;
        }
        let mut image_inputs=n.inputs.clone();
        if n.op==0x02 {if let Slot::Input(k)=n.param("opacity") {image_inputs.push(*k);}}
        for &k in &image_inputs {
            let size = self.gpu.as_ref().unwrap().size(k).or_else(|| self.images.get(k).and_then(|i| i.as_ref()).map(|im| (im.w, im.h)));
            let Some((iw, ih)) = size else { return false; };
            if !self.gpu.as_ref().unwrap().fits(iw, ih) {
                return false;
            }
            if (iw, ih) != (w, h) {
                if !matches!(n.op, 0x02 | 0x1e) {
                    if crate::envc::var("SDRT_GPU_LOG").is_ok() {
                        eprintln!("GPU {kern} node {}: CPU fallback (input size mismatch)", n.index);
                    }
                    return false;
                }
                const RESIZE_SWITCHES: [&str; 3] = ["SDRT_RESIZE", "SDRT_RESIZEMIP", "SDRT_BIL256"];
                if RESIZE_SWITCHES.iter().any(|s| crate::envc::var(s).is_ok()) {
                    if crate::envc::var("SDRT_GPU_LOG").is_ok() {
                        eprintln!("GPU {kern} node {}: CPU fallback (switch disabled)", n.index);
                    }
                    return false;
                }
            }
        }
        if image_inputs.len() > 2 {
            let g = self.gpu.as_ref().unwrap();
            let total = (image_inputs.len() as u64) * (w as u64) * (h as u64) * 16;
            if total > g.limits().max_storage_buffer_binding_size as u64 {
                return false;
            }
        }
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let gpu = self.gpu.as_mut().unwrap();
        for &k in &image_inputs {
            if gpu.size(k).is_none() {
                gpu.upload(k, self.images[k].as_ref().expect("CPU input computed")).expect("GPU upload failed");
            }
        }
        const GPU_RESIZE_BASE: usize = usize::MAX - 64;
        let mut resized_inputs: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut tmp_counter = 0;
        for &k in &image_inputs {
            let (iw, ih, icolor) = self.gpu_image_info(k).expect("input info");
            if (iw, ih) != (w, h) && !resized_inputs.contains_key(&k) {
                let tmp_id = GPU_RESIZE_BASE + tmp_counter;
                tmp_counter += 1;
                let fmt = self.tex_format(k);
                let mut rconfig = [0u32; 40];
                rconfig[0] = 7;
                rconfig[12..16].copy_from_slice(&[iw as u32, ih as u32, fmt, u32::from(icolor)]);
                let pass = crate::gpu::Pass {
                    config: rconfig,
                    w,
                    h,
                    color: icolor,
                    format: 3,
                    a: crate::gpu::Src::Node(k),
                    b: crate::gpu::Src::None,
                };
                self.gpu.as_mut().unwrap().dispatch_passes(tmp_id, crate::gpu::ATOMIC, &[pass]).expect("resize pass failed");
                resized_inputs.insert(k, tmp_id);
            }
        }
        let mapped_k = |k: usize| *resized_inputs.get(&k).unwrap_or(&k);
        let effective_inputs: Vec<usize> = image_inputs.iter().map(|&k| mapped_k(k)).collect();
        let mut config = [0u32; 40];
        let (mut a, mut b) = (None, None);
        let mut sources=Vec::new();
        match n.op {
            0x24=>{
                config[0]=5; config[3]=n.b3&3;
                let intensity=self.param(n,"intensity",seed).map_or(1.0,|v|v[0]);
                let inv=match n.param("inversedy") {
                    Slot::Const(v)=>v.first().is_some_and(|&x|x!=0),
                    Slot::Fn(_)=>self.param(n,"inversedy",seed).is_some_and(|v|v[0]!=0.0),
                    _=>false,
                };
                let height_alpha=self.p.version<5 || match n.param("input2alpha") {
                    Slot::Const(_)=>true,
                    Slot::Fn(_)=>self.param(n,"input2alpha",seed).is_some_and(|v|v[0]!=0.0),
                    _=>false,
                };
                if height_alpha {config[3]|=4;}
                let ps=self.pixelsize(n,seed);
                config[4]=(intensity*w as f32/256.0*ps.0).to_bits();
                config[5]=(intensity*h as f32/256.0*ps.1*if inv {-1.0} else {1.0}).to_bits();
                a=Some(mapped_k(n.inputs[0]));
            }
            0x1a=>{
                config[0]=6; config[3]=n.b3&3;
                config[4]=self.param(n,"intensity",seed).map_or(1.0,|v|v[0]).to_bits();
                let ps=self.pixelsize(n,seed);config[5]=ps.0.to_bits();config[6]=ps.1.to_bits();
                a=Some(mapped_k(n.inputs[0]));
            }
            0x1e=>{
                config[0]=3;config[3]=n.flags&0x400;
                let nc=if n.color {4} else {1};
                for (j,(name,default)) in [("inlow",0.0),("inhigh",1.0),("inmid",0.5),("outlow",0.0),("outhigh",1.0)].into_iter().enumerate() {
                    let v=self.param(n,name,seed).unwrap_or(vec![default;nc]);
                    for c in 0..4 {
                        let mut x=v[c%nc].clamp(0.0,1.0);
                        if j==2 {x=if x>=0.5 {1.0+16.0*(x-0.5)} else {1.0/(1.0+16.0*(0.5-x))};}
                        config[20+j*4+c]=x.to_bits();
                    }
                }
                a=Some(mapped_k(n.inputs[0]));
            }
            0x02=>{
                config[0]=4;config[3]=n.flags;
                let opacity=match n.param("opacity") {
                    Slot::Input(_)=>self.param(n,"opacitymult",seed).map_or(1.0,|v|v[0]),
                    s=>self.slot(&s.clone(),seed).map_or(1.0,|v|v[0]),
                };
                config[4]=opacity.to_bits();
                if let Slot::Const(v)=n.param("mask") {
                    if n.flags&0x40!=0 && v.len()==2 {
                        config[5]=1.0f32.to_bits();
                        let q=|u:u32,dim:usize|(((u&65535) as f32/32768.0)*dim as f32).floor() as u32;
                        config[12..16].copy_from_slice(&[q(v[0],w),q(v[0]>>16,h),q(v[1],w),q(v[1]>>16,h)]);
                    }
                }
                a=Some(mapped_k(n.inputs[0]));b=Some(mapped_k(n.inputs[1]));
                sources=effective_inputs.iter().map(|&k|(Some(k), if resized_inputs.values().any(|&v| v == k) { 3 } else { self.tex_format(k) })).collect();
            }
            0x0c => {
                let v = self.param(n, "color", seed);
                let c = if n.color { v.map_or([0., 0., 0., 1.], |v| [v[0], v[1], v[2], v[3]]) } else { Image::gray(v.map_or(0., |v| v[0])) };
                config[4..8].copy_from_slice(&c.map(f32::to_bits));
            }
            0x06 if n.color => {
                config[0] = 1;
                config[12..16].copy_from_slice(&n.flags.to_le_bytes().map(u32::from));
                a = Some(mapped_k(n.inputs[0]));
                b = Some(mapped_k(n.inputs[1]));
            }
            _ => {
                config[0] = 2;
                config[3] = u32::from(n.b3 & 16 != 0);
                let v = self.param(n, "weights", seed).unwrap_or(vec![1. / 3., 1. / 3., 1. / 3., 0.]);
                for c in 0..4 {
                    config[4 + c] = v[c].to_bits();
                }
                config[8] = self.param(n, "mattelevel", seed).map_or(1., |v| v[0]).to_bits();
                a = Some(mapped_k(n.inputs[0]));
            }
        }
        let gpu = self.gpu.as_mut().unwrap();
        gpu.dispatch(n.index, w, h, n.color, n.format(), &config, a, b, None, &sources).expect("Vulkan dispatch failed");
        for &tmp_id in resized_inputs.values() {
            self.gpu.as_mut().unwrap().release(tmp_id);
        }
        if crate::envc::var("SDRT_GPU_LOG").is_ok() {
            eprintln!("GPU {kern} node {}: shader {w}x{h}", n.index);
        }
        self.images[n.index] = None;
        true
    }

    fn exec(&mut self, n: &Node) -> Image {
        let (w, h) = n.size();
        self.cur_size = [w as f32, h as f32];
        let seed = self.seed(n);
        let i = n.index;
        match n.op {
            0x0A => self.svg(n, seed),
            0x22 => self.text(n, seed),
            0x0C => {
                let v = self.param(n, "color", seed);
                let c = if n.color {
                    v.map_or([0.0, 0.0, 0.0, 1.0], |v| [v[0], v[1], v[2], v[3]])
                } else {
                    Image::gray(v.map_or(0.0, |v| v[0]))
                };
                Image::new(w, h, n.color, c)
            }
            0x02 => self.blend(n, seed),
            0x1E => self.levels(n, seed),
            0x1C => self.hsl(n, seed),
            0x04 => self.transformation(n, seed),
            0x00 => self.gradient_map(n, seed),
            0x06 if n.color => {
                let (a, b) = (self.input_at(n.inputs[0], w, h), self.input_at(n.inputs[1], w, h));
                let sel = n.flags.to_le_bytes();
                let mut o = Image::new(w, h, true, [0.0; 4]);
                for k in 0..w * h {
                    let (pa, pb) = (a.px[k], b.px[k]);
                    for c in 0..4 {
                        let s = sel[c] as usize;
                        o.px[k][c] = if s < 4 { pa[s] } else { pb[(s - 4) & 3] };
                    }
                }
                o
            }
            0x06 => {
                let src = self.input_at(n.inputs[0], w, h);
                let wts = self.param(n, "weights", seed).unwrap_or(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]);
                let matte = self.param(n, "mattelevel", seed);
                let mut o = Image::new(w, h, false, [0.0; 4]);
                for k in 0..w * h {
                    let p = src.px[k];
                    let mut g = p[0] * wts[0] + p[1] * wts[1] + p[2] * wts[2] + p[3] * wts[3];
                    if n.b3 & 0x10 != 0 {
                        // alpha flattened against the matte level, default 1 (lab/py/grayprobe.py)
                        g = lerp(matte.as_ref().map_or(1.0, |m| m[0]), g, p[3]);
                    }
                    o.px[k] = Image::gray(g);
                }
                o
            }
            0x20 => self.bitmap(n),
            0x28 => self.pixel_processor(n, seed),
            0x08 => crate::fxmap::render(self, n, seed),
            0x24 => self.normal(n, seed),
            0x14 => self.blur(n, seed),
            0x0E => self.warp(n, seed),
            0x1A => self.sharpen(n, seed),
            0x2C => self.curve(n),
            0x2A => self.distance(n, seed),
            0x10 => self.emboss(n, seed),
            0x26 => self.dyn_gradient(n, seed),
            0x18 => self.directional_warp(n, seed),
            0x16 => self.motion_blur(n, seed),
            0x12 => self.legacy_motion_blur(n, seed),
            _ => {
                self.unsupported(i, format!("op {:#04x}", n.op));
                Image::new(w, h, n.color, if n.color { [0.0; 4] } else { Image::gray(0.0) })
            }
        }
    }

    fn blend(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let bg = self.input_at_fmt(n.inputs[0], w, h);
        let fg = self.input_at_fmt(n.inputs[1], w, h);
        let mode = n.flags & 0xF;
        let alpha_mode = (n.flags >> 11) & 3;
        let (op_img, op_const) = match n.param("opacity") {
            Slot::Input(k) => (Some(self.input_at_fmt(*k, w, h)), self.param(n, "opacitymult", seed).map_or(1.0, |v| v[0])),
            s => (None, self.slot(&s.clone(), seed).map_or(1.0, |v| v[0])),
        };
        // the Cropping area: (x0, y0) (x1, y1) as u16/32768; pixels floor(x0·w) ≤ x < floor(x1·w)
        // (same for y) blend, the others keep the background (lab/py/maskprobe.py)
        let crop = match n.param("mask") {
            Slot::Const(v) if n.flags & 0x40 != 0 && v.len() == 2 => {
                let q = |u: u32, dim: usize| ((u & 0xFFFF) as f32 / 32768.0 * dim as f32).floor() as i64;
                Some((q(v[0], w), q(v[0] >> 16, h), q(v[1], w), q(v[1] >> 16, h)))
            }
            _ => None,
        };
        // All modes and alpha modes measured against sbsrender (lab/py/blendtest.py, blendfit*.py).
        let f = |b: f32, f: f32| -> f32 {
            match mode {
                0 | 7 => f,
                1 => b + f,
                2 => b - f,
                3 => b * f,
                4 => b + 2.0 * f - 1.0,
                5 => b.max(f),
                6 => b.min(f),
                // 0/0 is 1 on the reference GPU (both versions; metalwalkwaysubstance002 705/1005,
                // all-zero inputs → 1.0: its clamp is min-first, and min(NaN, 1) = 1); SDRT_DIVNAN=0: NaN
                8 => { let d = b / f; if d.is_nan() && crate::env_once!("SDRT_DIVNAN").as_deref() != Ok("0") { 1.0 } else { d } }
                9 => if b < 0.5 { 2.0 * b * f } else { 1.0 - 2.0 * (1.0 - b) * (1.0 - f) },
                10 => 1.0 - (1.0 - b) * (1.0 - f),
                11 => if f < 0.5 { 2.0 * b * f + b * b * (1.0 - 2.0 * f) } else { 2.0 * b * (1.0 - f) + b.sqrt() * (2.0 * f - 1.0) },
                _ => f,
            }
        };
        // the GPU's mix: b·(1 − t) + f·t (m_brick_01 node 202: copy at opacity 0.5 of 3915 and
        // 36982 sixteen-bit codes, a tie at 20448.5 that b + (f − b)·t rounds up, the reference
        // down); SDRT_MIX=old for b + (f − b)·t
        let old_mix = crate::envc::var("SDRT_MIX").as_deref() == Ok("old");
        let lerp = |a: f32, b: f32, t: f32| if old_mix { a + (b - a) * t } else { a * (1.0 - t) + b * t };
        let fmt = n.format();
        let half_rne = crate::env_once!("SDRT_HALF").as_deref() == Ok("rne");
        let crop_q = crop.is_some() && crate::env_once!("SDRT_CROPQ").as_deref() != Ok("0");
        use rayon::prelude::*;
        let color = n.color;
        let mut px: Vec<[f32; 4]> = (0..w * h)
            .into_par_iter()
            .map(|k| {
            let (pb, pf) = (bg.px[k], fg.px[k]);
            if let Some((x0, y0, x1, y1)) = crop {
                let (x, y) = ((k % w) as i64, (k / w) as i64);
                if x < x0 || x >= x1 || y < y0 || y >= y1 {
                    return pb;
                }
            }
            // A cropped opacity mask is rendered into a temporary mask buffer of the blend node's
            // own format before multiplying by opacitymult (sovietfloortile 3096; probe_blend_crop.py).
            let op_m = op_img.as_ref().map_or(1.0, |im| {
                let m = im.px[k][0];
                if crop_q {
                    match fmt {
                        0 => crate::image::unorm(m, 8) as f32 / 255.0,
                        1 => crate::image::unorm(m, 16) as f32 / 65535.0,
                        2 if !half_rne => crate::image::half_trunc(m),
                        2 => crate::image::half(m),
                        _ => m,
                    }
                } else {
                    m
                }
            });
            // the opacity is clamped to [0, 1] (scratch/fb/blendop.py: 9, 3, 1.5 act as 1; −0.5 as 0)
            let op = (op_const * op_m).clamp(0.0, 1.0);
            if !color {
                return Image::gray(lerp(pb[0], f(pb[0], pf[0]), op));
            }
            let (ba, fa) = (pb[3], pf[3]);
            let mut c = [0.0; 4];
            if mode == 7 {
                // switch: every channel from background to foreground by the opacity
                for ch in 0..4 {
                    c[ch] = lerp(pb[ch], pf[ch], op);
                }
            } else {
                match alpha_mode {
                    1 => {
                        // ignore alpha
                        for ch in 0..3 {
                            c[ch] = lerp(pb[ch], f(pb[ch], pf[ch]), op);
                        }
                        c[3] = ba + op * fa * (1.0 - ba);
                    }
                    2 => {
                        // straight: W3C compositing of the two layers, mixed by the opacity
                        let a_full = fa + ba * (1.0 - fa);
                        let a = ba * (1.0 - op) + a_full * op;
                        for ch in 0..3 {
                            let mixed = f(pb[ch], pf[ch]) * fa * ba + pf[ch] * fa * (1.0 - ba) + pb[ch] * ba * (1.0 - fa);
                            let pre = pb[ch] * ba * (1.0 - op) + mixed * op;
                            c[ch] = if a != 0.0 { pre / a } else { 0.0 };
                        }
                        c[3] = a;
                    }
                    3 if mode == 0 => {
                        // premultiplied copy
                        for ch in 0..3 {
                            c[ch] = pf[ch] * op + pb[ch] * (1.0 - fa * op);
                        }
                        c[3] = fa * op + ba * (1.0 - fa * op);
                    }
                    _ => {
                        // source alpha (premultiplied modes other than copy behave the same)
                        for ch in 0..3 {
                            c[ch] = lerp(pb[ch], f(pb[ch], pf[ch]), fa * op);
                        }
                        c[3] = ba + op * fa * (1.0 - ba);
                    }
                }
            }
            c
            })
            .collect();
        // Half-float blend and levels nodes truncate toward zero on storage (cream-leather-01 138;
        // probe_half.py), just like 16F FX-Maps and gradient maps.
        if fmt == 2 && !half_rne {
            for p in &mut px {
                *p = if color { p.map(crate::image::half_trunc) } else { Image::gray(crate::image::half_trunc(p[0])) };
            }
        }
        Image { w, h, color, px }
    }

    fn levels(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let src = self.input_at_fmt(n.inputs[0], w, h);
        let nc = if n.color { 4 } else { 1 };
        let get = |rt: &mut Self, name: &str, d: f32| -> Vec<f32> { rt.param(n, name, seed).unwrap_or(vec![d; nc]) };
        let (il, ih, im, ol, oh) = (get(self, "inlow", 0.0), get(self, "inhigh", 1.0), get(self, "inmid", 0.5), get(self, "outlow", 0.0), get(self, "outhigh", 1.0));
        let clamp = n.flags & 0x400 == 0;
        // every parameter is clamped to [0, 1], whatever the clamp flag (redcryptfloor 304: outhigh
        // from a function = 1.1 acts as 1; scratch/fb/lvclamp.py). SDRT_LVPCLAMP=0 = old
        let pc = crate::envc::var("SDRT_LVPCLAMP").as_deref() != Ok("0");
        let c01 = |v: Vec<f32>| -> Vec<f32> { if pc { v.into_iter().map(|x| x.clamp(0.0, 1.0)).collect() } else { v } };
        let (il, ih, im, ol, oh) = (c01(il), c01(ih), c01(im), c01(ol), c01(oh));
        if crate::envc::var("SDRT_LVDBG").is_ok() { eprintln!("levels {} in {:?} {:?} mid {:?} out {:?} {:?}", n.index, il, ih, im, ol, oh); }
        let mut o = Image::new(w, h, n.color, [0.0; 4]);
        use rayon::prelude::*;
        o.px.par_iter_mut().enumerate().for_each(|(k, out)| {
            let p = src.px[k];
            let mut q = p;
            for c in 0..nc {
                // inlow == inhigh: the range becomes 1e-4 (a ramp 1e-4 wide; exactly at inlow 0 —
                // ceiling_orna_002's levels on black areas). Only an exactly empty range: ±5e-5
                // divide as they are (scratch/fb/lvrange.py; redbrickbuilding 1106: 0.5 + 0.5/65535
                // → 5000/65535)
                let d = if ih[c] != il[c] { ih[c] - il[c] } else { 1e-4 };
                let mut x = (p[c] - il[c]) / d;
                if clamp {
                    x = x.clamp(0.0, 1.0);
                }
                let m = im[c];
                let e = if m >= 0.5 { 1.0 + 16.0 * (m - 0.5) } else { 1.0 / (1.0 + 16.0 * (0.5 - m)) };
                if e != 1.0 {
                    x = x.max(0.0).powf(e);
                }
                let mut y = ol[c] + x * (oh[c] - ol[c]);
                if clamp {
                    y = y.clamp(0.0, 1.0);
                }
                q[c] = y;
            }
            *out = if n.color { q } else { Image::gray(q[0]) };
        });
        if n.format() == 2 && crate::env_once!("SDRT_HALF").as_deref() != Ok("rne") {
            for p in &mut o.px {
                *p = if n.color { p.map(crate::image::half_trunc) } else { Image::gray(crate::image::half_trunc(p[0])) };
            }
        }
        o
    }

    fn hsl(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let src = self.input_at(n.inputs[0], w, h);
        let hue = self.param(n, "hue", seed).map_or(0.5, |v| v[0]);
        let sat = self.param(n, "saturation", seed).map_or(0.5, |v| v[0]);
        let lum = self.param(n, "luminosity", seed).map_or(0.5, |v| v[0]);
        let mut o = Image::new(w, h, true, [0.0; 4]);
        use rayon::prelude::*;
        o.px.par_iter_mut().enumerate().for_each(|(k, out)| {
            let p = src.px[k];
            let (mut hh, mut s, mut l) = rgb_to_hsl(p[0], p[1], p[2]);
            hh = (hh + (hue - 0.5) * 2.0).rem_euclid(1.0);
            s = (s + 2.0 * (sat - 0.5)).clamp(0.0, 1.0);
            l = (l + 2.0 * (lum - 0.5)).clamp(0.0, 1.0);
            let (r, g, b) = hsl_to_rgb(hh, s, l);
            *out = [r, g, b, p[3]];
        });
        o
    }

    /// Mip levels 1.. of node k's image (cached; level 0 is the image itself).
    fn mip_levels(&mut self, k: usize) -> std::sync::Arc<Vec<Image>> {
        // an embedded bitmap's mip levels keep full precision (lizard_scales node 3: 1024² JPEG →
        // 256², exact only so); other nodes store each level in their format
        let node = &self.p.nodes[k];
        let fmt = match crate::envc::var("SDRT_MIPFLOAT").as_deref() {
            Ok("all") => 3,
            Ok("half") => 2,
            Ok("16") => 1,
            Ok("none") => node.format(),
            _ if node.op & 0xFE == 0x20 => 3,
            _ => node.format(),
        };
        self.mip_levels_fmt(k, fmt)
    }

    fn mip_levels_fmt(&mut self, k: usize, fmt: u32) -> std::sync::Arc<Vec<Image>> {
        if let Some(c) = self.mip_cache.get(&(k, fmt)) {
            return c.clone();
        }
        let mut chain = crate::image::transform_mips(self.img(k), fmt, self.p.nodes[k].format());
        chain.remove(0);
        let rc = std::sync::Arc::new(chain);
        self.mip_cache.insert((k, fmt), rc.clone());
        rc
    }

    /// SVG (op 0x0a): a command stream of words (low 3 bits command, the rest n): 3 set color
    /// (RGB24; gray = the channel mean), 1 fill: n points (u16 x, y / 65536) as a **triangle strip**,
    /// 2 stroke n points, 4/5 gradients (unused in the corpus), 0 end. No anti-aliasing: a pixel is
    /// covered when its centre is inside a triangle (scratch/fb/svgprobe.py: rectangle, triangle,
    /// concave strip and a 0.6-px bar exact). Background: the color param (byte3 bit 1), else
    /// transparent black. Strokes: 1-px lines by the diamond-exit rule. TODO: background input
    /// (byte3 bit 2, unused in the corpus), edge tie rules, gradients (unused).
    fn svg(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let p = self.p;
        let bgc = self.param(n, "color", seed);
        let bg = match (&bgc, n.color) {
            (Some(v), true) => [v[0], v[1], v[2], v.get(3).copied().unwrap_or(1.0)],
            (Some(v), false) => Image::gray(v[0]),
            (None, true) => [0.0; 4],
            (None, false) => Image::gray(0.0),
        };
        let mut img = match n.inputs.first() { Some(&k) => self.input_at(k, w, h).as_ref().clone(), None => Image::new(w, h, n.color, bg) };
        let mut o = n.data;
        let mut color = [1.0f32, 1.0, 1.0, 1.0];
        // vertices snap to 1/256 px (8 sub-pixel bits) and a centre exactly on an edge counts for
        // top-left edges (variant 1): corpus SVG nodes 12 of 42 failing ones exact, total error ÷4
        let svg_tie: u32 = crate::envc::var("SDRT_SVGTIE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let svg_snap: f64 = crate::envc::var("SDRT_SVGSNAP").ok().and_then(|v| v.parse().ok()).unwrap_or(256.0);
        for _ in 0..1_000_000 {
            let wd = p.w(o); o += 4;
            let (cmd, cnt) = (wd & 7, (wd >> 3) as usize);
            match cmd {
                0 => break,
                // the low byte is red (geometric_flowers_v2 5496…7968: yellow/red where the
                // high-byte-red order gave cyan/blue); SDRT_SVGRGB=1: the old order
                3 if crate::env_once!("SDRT_SVGRGB").is_ok() => color = [((cnt >> 16) & 255) as f32 / 255.0, ((cnt >> 8) & 255) as f32 / 255.0, (cnt & 255) as f32 / 255.0, 1.0],
                3 => color = [(cnt & 255) as f32 / 255.0, ((cnt >> 8) & 255) as f32 / 255.0, ((cnt >> 16) & 255) as f32 / 255.0, 1.0],
                1 | 2 => {
                    // a strip lies inside the program (malformed counts would allocate gigabytes)
                    let cnt = cnt.min(p.b.len() / 4);
                    let snap = svg_snap;
                    // SDRT_SVGOFF=dx,dy: a vertex offset in pixels (test switch)
                    let off: (f64, f64) = crate::envc::var("SDRT_SVGOFF").ok().and_then(|s| { let v: Vec<f64> = s.split(',').filter_map(|t| t.parse().ok()).collect(); (v.len() == 2).then(|| (v[0], v[1])) }).unwrap_or((0.0, 0.0));
                    let pts: Vec<(f64, f64)> = (0..cnt).map(|i| {
                        let v = p.w(o + 4 * i);
                        let (x, y) = ((v & 0xFFFF) as f64 / 65536.0 + off.0 / w as f64, (v >> 16) as f64 / 65536.0 + off.1 / h as f64);
                        // GPU vertex snapping to 1/snap pixel (SDRT_SVGSNAP; 0 = none)
                        if snap > 0.0 { ((x * w as f64 * snap).round() / (snap * w as f64), (y * h as f64 * snap).round() / (snap * h as f64)) } else { (x, y) }
                    }).collect();
                    o += 4 * cnt;
                    let c = if n.color { color } else { Image::gray((color[0] + color[1] + color[2]) / 3.0) };
                    if cmd == 2 {
                        // stroke: a 1-px aliased line strip, OpenGL's diamond-exit rule: a pixel is lit
                        // when the segment crosses its diamond |dx| + |dy| < ½ and does not end inside
                        // it (svgprobe.py strokes: horizontal, vertical, diagonal, polyline exact)
                        let pp: Vec<(f64, f64)> = pts.iter().map(|&(x, y)| (x * w as f64, y * h as f64)).collect();
                        for sg in pp.windows(2) {
                            let (a, b) = (sg[0], sg[1]);
                            let min_l1 = |cx: f64, cy: f64| -> f64 {
                                let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                                let mut ts = vec![0.0, 1.0];
                                if dx != 0.0 { ts.push((cx - a.0) / dx); }
                                if dy != 0.0 { ts.push((cy - a.1) / dy); }
                                ts.into_iter().filter(|t| (0.0..=1.0).contains(t))
                                    .map(|t| ((a.0 + t * dx) - cx).abs() + ((a.1 + t * dy) - cy).abs()).fold(f64::INFINITY, f64::min)
                            };
                            let (x0, x1) = ((a.0.min(b.0) - 1.0).floor().max(0.0) as usize, ((a.0.max(b.0) + 1.0).ceil() as usize).min(w - 1));
                            let (y0, y1) = ((a.1.min(b.1) - 1.0).floor().max(0.0) as usize, ((a.1.max(b.1) + 1.0).ceil() as usize).min(h - 1));
                            for py in y0..=y1 {
                                for px in x0..=x1 {
                                    let (cx, cy) = (px as f64 + 0.5, py as f64 + 0.5);
                                    if min_l1(cx, cy) < 0.5 && !((b.0 - cx).abs() + (b.1 - cy).abs() < 0.5) {
                                        img.px[py * w + px] = c;
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    for t in pts.windows(3) {
                        let (a, b, cc) = (t[0], t[1], t[2]);
                        // zero-area triangles (repeated points join the strip's pieces) cover nothing
                        if (b.0 - a.0) * (cc.1 - a.1) - (b.1 - a.1) * (cc.0 - a.0) == 0.0 { continue; }
                        let (x0, x1) = (a.0.min(b.0).min(cc.0), a.0.max(b.0).max(cc.0));
                        let (y0, y1) = (a.1.min(b.1).min(cc.1), a.1.max(b.1).max(cc.1));
                        let px0 = ((x0 * w as f64 - 0.5).floor().max(0.0)) as usize;
                        let px1 = ((x1 * w as f64 - 0.5).ceil().min(w as f64 - 1.0)).max(0.0) as usize;
                        let py0 = ((y0 * h as f64 - 0.5).floor().max(0.0)) as usize;
                        let py1 = ((y1 * h as f64 - 0.5).ceil().min(h as f64 - 1.0)).max(0.0) as usize;
                        let e = |p: (f64, f64), q: (f64, f64), x: f64, y: f64| (q.0 - p.0) * (y - p.1) - (q.1 - p.1) * (x - p.0);
                        // orient so that the interior gives positive edge functions; centres exactly on
                        // an edge count for "top-left" edges only (SDRT_SVGTIE picks the convention:
                        // 0 inclusive, 1–4 sign variants)
                        let (a, b, cc) = if e(a, b, cc.0, cc.1) < 0.0 { (a, cc, b) } else { (a, b, cc) };
                        let tie = svg_tie;
                        let owns = |p: (f64, f64), q: (f64, f64)| -> bool {
                            let (dx, dy) = (q.0 - p.0, q.1 - p.1);
                            match tie { 0 => true, 1 => dy < 0.0 || (dy == 0.0 && dx > 0.0), 2 => dy > 0.0 || (dy == 0.0 && dx < 0.0),
                                3 => dy < 0.0 || (dy == 0.0 && dx < 0.0), _ => dy > 0.0 || (dy == 0.0 && dx > 0.0) }
                        };
                        let (o1, o2, o3) = (owns(a, b), owns(b, cc), owns(cc, a));
                        for py in py0..=py1 {
                            for px in px0..=px1 {
                                let (x, y) = ((px as f64 + 0.5) / w as f64, (py as f64 + 0.5) / h as f64);
                                let (d1, d2, d3) = (e(a, b, x, y), e(b, cc, x, y), e(cc, a, x, y));
                                let inside = |d: f64, own: bool| d > 0.0 || (d == 0.0 && own);
                                if inside(d1, o1) && inside(d2, o2) && inside(d3, o3) { img.px[py * w + px] = c; }
                            }
                        }
                    }
                }
                4 => o += 8 * cnt,
                5 => o += 4 * cnt,
                _ => { self.unsupported(n.index, "SVG command"); break; }
            }
        }
        img
    }

    /// Text (op 0x22): the string (block offset data: u32 n, n UTF-32 code points) in the embedded
    /// font (data2: u32 id, u32 length, the TrueType file), anti-aliased. Layout model under test:
    /// baseline anchor at (½, ½) + position, em = fontsize·K (SDRT_TEXTK, uv units), alignment
    /// 0 left / 1 centre / 2 right of the advance width. TODO: matrix, fit K, the AA rule, CFF fonts.
    fn text(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let p = self.p;
        let bg = if n.color { [0.0; 4] } else { Image::gray(0.0) };
        let mut img = match n.inputs.first() { Some(&k) => self.input_at(k, w, h).as_ref().clone(), None => Image::new(w, h, n.color, bg) };
        // flags bits 0-1: 0 the string in the block (u32 count, UTF-32), 2 a string input's uid
        let cps: Vec<u32> = if n.flags & 3 == 2 {
            match self.strings.get(&(n.data as u32)) {
                Some(s) => s.chars().map(|c| c as u32).collect(),
                None => { self.unsupported(n.index, "text from an unknown string input"); return img; }
            }
        } else {
            let cnt = p.w(n.data) as usize;
            (0..cnt.min(4096)).map(|i| p.w(n.data + 4 + 4 * i)).collect()
        };
        if n.data2 == 0 { self.unsupported(n.index, "text without a font"); return img; }
        let flen = p.w(n.data2 + 4) as usize;
        let fdata = p.bytes(n.data2 + 8, flen);
        let Some(font) = crate::text::Font::parse(fdata) else { self.unsupported(n.index, "text font (not TrueType)"); return img };
        let color = self.param(n, "color", seed).unwrap_or(vec![1.0; 4]);
        let pos = self.param(n, "position", seed).unwrap_or(vec![0.0, 0.0]);
        // the matrix maps an output offset from the anchor (0.5 + position) to text space,
        // column-major as the transformation node's: t = (m0 dx + m2 dy, m1 dx + m3 dy)
        // (textprobe: [2,0,0,1] halves the width, [0,1,−1,0] turns the text 90° anticlockwise);
        // outlines go the other way, d = M⁻¹ t
        let mat = self.param(n, "matrix", seed).unwrap_or(vec![1.0, 0.0, 0.0, 1.0]);
        let (m0, m1, m2, m3) = (mat[0] as f64, mat[1] as f64, mat[2] as f64, mat[3] as f64);
        let det = m0 * m3 - m2 * m1;
        let inv = if det.abs() > 1e-12 { [m3 / det, -m1 / det, -m2 / det, m0 / det] } else { [1.0, 0.0, 0.0, 1.0] };
        let _ident = mat[..4] == [1.0, 0.0, 0.0, 1.0];
        let piv = (0.5 + pos[0] as f64, 0.5 + pos[1] as f64);
        let to_out = move |x: f64, y: f64| -> (f64, f64) {
            let (dx, dy) = (x - piv.0, y - piv.1);
            (piv.0 + inv[0] * dx + inv[2] * dy, piv.1 + inv[1] * dx + inv[3] * dy)
        };
        let fsize = self.param(n, "fontsize", seed).map_or(0.25, |v| v[0]);   // default measured (textprobe)
        // alignment 0 left, 1 centre, 2 right; 3: from the "alignment" function
        let align = match (n.flags >> 12) & 3 {
            3 => self.param(n, "alignment", seed).map_or(0, |v| (v[0].round() as i64).clamp(0, 2) as u32),
            a => a,
        };
        // the font size is the line height ascent − descent (hhea), not the em (checker_1m 427)
        let k: f64 = crate::envc::var("SDRT_TEXTK").ok().and_then(|v| v.parse().ok()).unwrap_or(font.upem / (font.ascent - font.descent));
        if crate::envc::var("SDRT_TEXTDBG").is_ok() { eprintln!("text {} upem {} ascent {} descent {} k {} fsize {} pos {:?} align {} string {:?}", n.index, font.upem, font.ascent, font.descent, k, fsize, pos, align, cps.iter().filter_map(|&c| char::from_u32(c)).collect::<String>()); }
        let em = fsize as f64 * k;                     // uv per em
        let sc = em / font.upem;
        // lines at '\n', each aligned on its own; the baseline steps down by the font's line height
        // ascent − descent + lineGap of hhea (scratch/fb/textprobe.py: 52.70 px at fontsize 0.2 in
        // 256², Liberation Sans 1854/−434/67; hhea lineGap 1000 → 73.2; OS/2 metrics ignored)
        let lh = crate::envc::var("SDRT_TEXTLH").ok().and_then(|v| v.parse::<f64>().ok())
            .unwrap_or((font.ascent - font.descent + font.line_gap) / (font.ascent - font.descent)) * fsize as f64;
        // SDRT_TEXTSNAP: 0 none (default: textprobe glyph edges exact without, the old +½ px only offset missing kerning), 1 start x + ½ px, 2 start rounded to a pixel, 3 each glyph origin
        // rounded, 4 start floored, 5 baseline rounded too
        let snap: u32 = crate::envc::var("SDRT_TEXTSNAP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        // segments per quadratic: SDRT_TEXTSTEPS (0 = adaptive bisection, see text::flatten)
        let steps: usize = crate::envc::var("SDRT_TEXTSTEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let r26 = crate::envc::var("SDRT_TEXT26").as_deref() == Ok("1");
        // SDRT_TEXTFLAT=px: the pixel-space bisection of text::flatten (steps, SDRT_TEXTFT) instead
        let stb = crate::envc::var("SDRT_TEXTFLAT").as_deref() != Ok("px");
        // all lines go through one rasterization, then one box clips them (textprobe made fonts):
        // rows floor(first line's ascender line) to floor(last line's descender line), columns
        // floor(left) to ceil(right) of the widest line's ink box — a narrower line reaching past it
        // is cut ('P\nB' right-aligned: P stops at ceil of B's right edge), a descender crossing into
        // the next line is not ('D\nP'). SDRT_TEXTCLIP=0 off.
        // one polygon set per glyph: each glyph is rasterized on its own and laid "over" the others
        let mut glyph_polys: Vec<Vec<Vec<(f64, f64)>>> = vec![];
        let nlines = cps.split(|&c| c == 10).count();
        let (mut lo_all, mut hi_all) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut wide, mut wide_adv_box) = (f64::NEG_INFINITY, (0.0f64, 0.0f64));
        let align_adv = crate::envc::var("SDRT_TEXTALIGN").as_deref() == Ok("adv");
        let (mut top, mut bot) = (0.0f64, 0.0f64);
        for (li, line) in cps.split(|&c| c == 10).enumerate() {
            let gids: Vec<usize> = line.iter().filter(|&&c| c != 13).map(|&c| font.glyph_index(c)).collect();
            // pair kerning ('kern' table) moves each glyph after the first; SDRT_TEXTKERN: 0 off,
            // 1 exact (default; textprobe AV/AVAV/LT AY at 128² and 256²: rounded pixels 0.023 vs 0.0008), 2 rounded to output pixels, 3 rounded to 1/64 px
            let kmode: u32 = crate::envc::var("SDRT_TEXTKERN").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            let kern = |a: usize, b: usize| -> f64 {
                let kx = font.kern(a, b) * sc * w as f64;   // output pixels
                (match kmode { 0 => 0.0, 1 => kx, 3 => (kx * 64.0).round() / 64.0, _ => kx.round() }) / (sc * w as f64)
            };
            let kerns: Vec<f64> = (0..gids.len()).map(|i| if i == 0 { 0.0 } else { kern(gids[i - 1], gids[i]) }).collect();
            // centre and right alignment move the line by half or all of its ink width, the union
            // of the glyphs' bounding boxes (an empty glyph, a space, a zero-width box at its pen)
            // from the first xMin to the last pen + xMax, with the pen origin (not the ink) at
            // anchor − width (textprobe made glyphs: 'I' with ink 100..400 centred moves by −150
            // units, 'IJ' right-aligned by −1800, 'sB' right-aligned ends at the anchor like 'B').
            // SDRT_TEXTALIGN=adv: the sum of advances
            let (lo, hi, has_bounds) = {
                let (mut pen, mut lo, mut hi) = (0.0, f64::INFINITY, f64::NEG_INFINITY);
                for (gi, &g) in gids.iter().enumerate() {
                    pen += kerns[gi];
                    let (a, b) = font.xbounds(g).unwrap_or((0.0, 0.0));
                    lo = lo.min(pen + a); hi = hi.max(pen + b);
                    pen += font.advance(g);
                }
                if hi >= lo { (lo, hi, true) } else { (0.0, 0.0, false) }
            };
            if has_bounds { lo_all = lo_all.min(lo); hi_all = hi_all.max(hi); }
            let total = if align_adv {
                (gids.iter().map(|&g| font.advance(g)).sum::<f64>() + kerns.iter().sum::<f64>()) * sc
            } else { (hi - lo) * sc };
            let ax = 0.5 + pos[0] as f64 - match align { 1 => total / 2.0, 2 => total, _ => 0.0 };
            let ay = 0.5 + pos[1] as f64 + li as f64 * lh;
            let ax = match snap { 1 => ax + 0.5 / w as f64, 2 | 5 => (ax * w as f64).round() / w as f64, 4 => (ax * w as f64).floor() / w as f64, _ => ax };
            let ay = if snap == 5 { (ay * h as f64).round() / h as f64 } else { ay };
            if total > wide { wide = total; wide_adv_box = (ax + lo * sc, ax + hi * sc); }
            if li == 0 { top = ay - font.ascent * sc; }
            if li + 1 == nlines { bot = ay - font.descent * sc; }
            let mut pen = 0.0;
            for (gi, &g) in gids.iter().enumerate() {
                pen += kerns[gi];
                let gx = ax + pen * sc;
                let gx = if snap == 3 { (gx * w as f64).round() / w as f64 } else { gx };
                let place = |x: f64, y: f64| -> (f64, f64) {
                    let (u, v) = to_out(gx + x * sc, ay - y * sc);
                    let (u, v) = (u * w as f64, v * h as f64);
                    // SDRT_TEXT26=1: points on FreeType's 26.6 grid (1/64 px)
                    if r26 { ((u * 64.0).round() / 64.0, (v * 64.0).round() / 64.0) } else { (u, v) }
                };
                let outline = font.outline(g, 0);
                let mut polys = vec![];
                if stb {
                    // flattened in font units (stb_truetype: flatness 0.35 px over min(scale_x, scale_y)),
                    // where the pixel-space row norms of the inverse matrix give the effective x/y scales
                    // (foliage 1174/1963 matrix [2.103,0,0,3.998], swater-gen 164 [0.99,0,0,0.78], tp_mat2
                    // shears/rotations/magnifications all exact). SDRT_TEXTFLATM=1: old unscaled/√|det|
                    let mat_px = if crate::envc::var("SDRT_TEXTFLATM").as_deref() == Ok("1") {
                        (w.min(h) as f64) / det.abs().sqrt().max(1e-12)
                    } else {
                        (w as f64 * inv[0].hypot(inv[2])).min(h as f64 * inv[1].hypot(inv[3])).max(1e-12)
                    };
                    let px_per_unit = sc * mat_px
                        * crate::envc::var("SDRT_TEXTFLATK").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0);
                    let flat = 0.35 / px_per_unit;
                    // CFF fonts: Type 2 charstrings, stb's cubic flattening
                    let flattened = font.flatten_cff(g, flat).unwrap_or_else(|| crate::text::flatten_stb(&outline, flat));
                    for pl in flattened {
                        polys.push(pl.into_iter().map(|(x, y)| place(x, y)).collect());
                    }
                } else {
                    let contours: Vec<Vec<(f64, f64, bool)>> = outline.into_iter()
                        .map(|c| c.into_iter().map(|(x, y, on)| { let (u, v) = place(x, y); (u, v, on) }).collect()).collect();
                    if crate::envc::var("SDRT_TEXTDBG").as_deref() == Ok("2") {
                        for c in &contours { eprintln!("contour {:?}", c.iter().map(|&(x, y, on)| format!("{:.2},{:.2}{}", x, y, if on { "" } else { "*" })).collect::<Vec<_>>()); }
                    }
                    polys.extend(crate::text::flatten(&contours, steps));
                }
                glyph_polys.push(polys);
                pen += font.advance(g);
            }
        }
        // tiling (byte 3 bit 0 x, bit 1 y) wraps glyphs across the edges, the box with them
        // (textprobe: 'Wide line' off the left edge reappears at the right with bit 0 only)
        let (tx, ty) = (n.b3 & 1 != 0, n.b3 & 2 != 0);
        // glyphs combine "over" (a + b − ab), each glyph's coverage a byte (textprobe: a half-pixel
        // rectangle stacked 1–4 times gives 127, 191, 223, 239 /255 = 1 − (1 − 127/255)ⁿ; 'll' at
        // small sizes). SDRT_TEXTCOMP=sum: one rasterization of everything (old)
        let comp_sum = crate::envc::var("SDRT_TEXTCOMP").as_deref() == Ok("sum");
        let do_clip = crate::envc::var("SDRT_TEXTCLIP").as_deref() != Ok("0");
        let clip_box = do_clip.then(|| {
            // whole-text horizontal clip box uses the union (lo_all, hi_all) of all non-empty
            // lines' unaligned ink bounds, aligned as one block (tp_clip probes 0–4: left-aligned
            // "0123\n...\np" in foliage 1174 keeps 'p''s left serif, right-aligned "Hq\nWide line"
            // clips at anchor + min_lo); with a matrix its four corners are mapped to the output
            let wide_box = if align_adv || lo_all > hi_all {
                wide_adv_box
            } else {
                let w_all = (hi_all - lo_all) * sc;
                let box_ax = 0.5 + pos[0] as f64 - match align { 1 => w_all / 2.0, 2 => w_all, _ => 0.0 };
                let box_ax = match snap { 1 => box_ax + 0.5 / w as f64, 2 | 5 => (box_ax * w as f64).round() / w as f64, 4 => (box_ax * w as f64).floor() / w as f64, _ => box_ax };
                (box_ax + lo_all * sc, box_ax + hi_all * sc)
            };
            let corners = [to_out(wide_box.0, top), to_out(wide_box.1, top), to_out(wide_box.0, bot), to_out(wide_box.1, bot)];
            let (bx0, bx1, by0, by1) = corners.iter().fold((f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
                |a, p| (a.0.min(p.0), a.1.max(p.0), a.2.min(p.1), a.3.max(p.1)));
            let (r0, r1) = ((by0 * h as f64).floor() as i64, (by1 * h as f64).floor() as i64);
            let (c0, c1) = ((bx0 * w as f64).floor() as i64, (bx1 * w as f64).ceil() as i64 - 1);
            (c0, c1, r0, r1, by0, by1)
        });
        let mut cov = vec![0f32; w * h];
        let mut all: Vec<Vec<(f64, f64)>> = vec![];
        for gp in glyph_polys {
            // tiling (byte 3 bit 0 x, bit 1 y) wraps glyphs across the edges, the box with them
            // (textprobe: 'Wide line' off the left edge reappears at the right with bit 0 only)
            let mut copies: Vec<((i64, i64), Vec<Vec<(f64, f64)>>)> = vec![((0, 0), gp.clone())];
            if tx || ty {
                let (x0, x1, y0, y1) = gp.iter().flatten().fold((f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
                    |a, p| (a.0.min(p.0), a.1.max(p.0), a.2.min(p.1), a.3.max(p.1)));
                for dy in [-1i32, 0, 1] {
                    for dx in [-1i32, 0, 1] {
                        if (dx == 0 && dy == 0) || (dx != 0 && !tx) || (dy != 0 && !ty) { continue; }
                        let (oxi, oyi) = (dx as i64 * w as i64, dy as i64 * h as i64);
                        let (ox, oy) = (oxi as f64, oyi as f64);
                        if x1 + ox <= 0.0 || x0 + ox >= w as f64 || y1 + oy <= 0.0 || y0 + oy >= h as f64 { continue; }
                        copies.push(((oxi, oyi), gp.iter().map(|pl| pl.iter().map(|p| (p.0 + ox, p.1 + oy)).collect()).collect()));
                    }
                }
            }
            if comp_sum { all.extend(copies.into_iter().map(|(_, c)| c).flatten()); continue; }
            for ((oxi, oyi), c) in copies {
                let (x0, x1, y0, y1) = c.iter().flatten().fold((f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
                    |a, p| (a.0.min(p.0), a.1.max(p.0), a.2.min(p.1), a.3.max(p.1)));
                if !(x0 < x1 && y0 < y1) { continue; }
                let (cx0, cx1, cy0, cy1) = match clip_box {
                    Some((c0, c1, r0, r1, _, _)) => ((c0 + oxi).max(0), (c1 + 1 + oxi).min(w as i64), (r0 + oyi).max(0), (r1 + 1 + oyi).min(h as i64)),
                    None => (0, w as i64, 0, h as i64),
                };
                let (bx0, by0) = ((x0.floor() as i64).max(cx0), (y0.floor() as i64).max(cy0));
                let (bx1, by1) = ((x1.ceil() as i64 + 1).min(cx1), (y1.ceil() as i64 + 1).min(cy1));
                if bx1 <= bx0 || by1 <= by0 { continue; }
                let (bw, bh) = ((bx1 - bx0) as usize, (by1 - by0) as usize);
                let local: Vec<Vec<(f64, f64)>> = c.iter().map(|pl| pl.iter().map(|p| (p.0 - bx0 as f64, p.1 - by0 as f64)).collect()).collect();
                let gc = crate::text::rasterize(&local, bw, bh);
                for yy in 0..bh {
                    for xx in 0..bw {
                        let a = gc[yy * bw + xx];
                        if a <= 0.0 { continue; }
                        let a = ((a * 255.0 + 0.5).floor().min(255.0)) / 255.0;   // stb: (int)(|k|·255 + ½)
                        let o = &mut cov[(by0 as usize + yy) * w + bx0 as usize + xx];
                        *o = *o + a * (1.0 - *o);
                    }
                }
            }
        }
        if comp_sum { cov = crate::text::rasterize(&all, w, h); }
        if let Some((c0, c1, r0, r1, top, bot)) = clip_box {
            if comp_sum {
                let inside = |v: i64, lo: i64, hi: i64, n: usize, wrap: bool| {
                    (v >= lo && v <= hi) || (wrap && ((v - n as i64 >= lo && v - n as i64 <= hi) || (v + n as i64 >= lo && v + n as i64 <= hi)))
                };
                for y in 0..h {
                    let iny = inside(y as i64, r0, r1, h, ty);
                    for x in 0..w {
                        if !(iny && inside(x as i64, c0, c1, w, tx)) { cov[y * w + x] = 0.0; }
                    }
                }
            }
            // 11.3 reference bug (fixed in 14.1: cream-leather-01 1029..1063, led_stripe 1578):
            // a box 1 or 2 px wide comes out with its rows spread apart (4-byte row alignment of a
            // power-of-two texture). Default off for 14.1; SDRT_TEXTNARROW=1 restores 11.3's bug.
            let bw = c1 - c0 + 1;
            if (bw == 1 || bw == 2) && crate::envc::var("SDRT_TEXTNARROW").as_deref() == Ok("1") {
                let bh = (r1 - r0 + 1).max(1) as usize;
                // kept rows: next_pow2(ceil(span) + 1), span = descender − ascender line in pixels
                // ('i\ni' at 0.03: span 7.79 keeps 16; led_stripe 1578 'R' span 2.56 keeps 4; made
                // fonts span 12.8/25.6/38.4 keep 16/32/64)
                let hp = (((bot - top) * h as f64).ceil() as usize + 1).next_power_of_two();
                let k = (4 / bw) as usize;
                let wrapi = |v: i64, n: usize| v.rem_euclid(n as i64) as usize;
                let at = |v: i64, n: usize, wrap: bool| if wrap { Some(wrapi(v, n)) } else if v >= 0 && v < n as i64 { Some(v as usize) } else { None };
                let src: Vec<Vec<f32>> = (0..bh).map(|j| (0..bw).map(|i| match (at(r0 + j as i64, h, ty), at(c0 + i, w, tx)) {
                    (Some(y), Some(x)) => cov[y * w + x], _ => 0.0 }).collect()).collect();
                for j in 0..bh { for i in 0..bw { if let (Some(y), Some(x)) = (at(r0 + j as i64, h, ty), at(c0 + i, w, tx)) { cov[y * w + x] = 0.0; } } }
                for d in 0..hp {
                    if d % k != 0 || d / k >= bh { continue; }
                    let Some(y) = at(r0 + d as i64, h, ty) else { continue };
                    for i in 0..bw { if let Some(x) = at(c0 + i, w, tx) { cov[y * w + x] = src[d / k][i as usize]; } }
                }
            }
        }
        for (k, &a) in cov.iter().enumerate() {
            if a <= 0.0 { continue; }
            let d = img.px[k];
            img.px[k] = if n.color {
                std::array::from_fn(|c| d[c] + (color[c.min(color.len() - 1)] - d[c]) * a)
            } else { Image::gray(d[0] + (color[0] - d[0]) * a) };
        }
        img
    }

    fn transformation(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let k = n.inputs[0];
        let m = self.param(n, "matrix", seed).unwrap_or(vec![1.0, 0.0, 0.0, 1.0]);
        let off = self.param(n, "offset", seed).unwrap_or(vec![0.0, 0.0]);
        let matte = self.param(n, "mattecolor", seed).unwrap_or(vec![0.0, 0.0, 0.0, 0.0]);
        let tiling = n.b3 & 3;
        let wrap = (tiling & 1 != 0, tiling & 2 != 0);
        let bilinear = n.flags & 0x20 != 0;
        let mip = n.flags & 0x1F;
        // mip: 0x1f automatic, 0x1e from a function, else a manual level.
        // Measured (lab/py/xfprobe.py): the matrix maps output to source directly, column-major,
        // and the offset adds in source space: uv_src = (m0 dx + m2 dy, m1 dx + m3 dy) + 0.5 + offset.
        let manual = match mip {
            0x1F => None,
            0x1E => Some(self.param(n, "miplevel", seed).map_or(0.0, |v| v[0])),
            m => Some(m as f32),
        };
        let (sw, sh) = (self.img(k).w, self.img(k).h);
        if crate::envc::var("SDRT_STRICT_TRACE").is_ok() {
            eprintln!("STRICT {{\"k\":{},\"op\":4,\"input\":{k},\"sfmt\":{},\"fmt\":{},\"sw\":{sw},\"sh\":{sh},\"w\":{w},\"h\":{h},\"flags\":{},\"tiling\":{tiling},\"matrix\":{m:?},\"offset\":{off:?},\"manual\":{}}}", n.index, self.p.nodes[k].format(), n.format(), n.flags, manual.map_or("null".into(), |v| v.to_string()));
        }
        // Auto-mip (mip = 0x1F) with a signed-permutation matrix ([±1,0,0,±1] or [0,±1,±1,0]) and
        // zero offset resizes per axis via mip halvings / 2× bilinear upsamples and flips/rotates
        // texels (wallpaper 406 m=[-1,0,0,-1], xf_all_probe/xf_ns). Manual mip levels (including
        // manual 0, flags=0x20: green-mosaic 98, pine_wood_panel_v2 284) sample the chosen level
        // directly without the auto-mip / 2×-upsample chain.
        let identity = m[..4] == [1.0, 0.0, 0.0, 1.0] && off[..2] == [0.0, 0.0];
        let dihedral = off[..2] == [0.0, 0.0]
            && ((m[0].abs() == 1.0 && m[1] == 0.0 && m[2] == 0.0 && m[3].abs() == 1.0)
                || (m[0] == 0.0 && m[1].abs() == 1.0 && m[2].abs() == 1.0 && m[3] == 0.0));
        let ident_mode = crate::envc::var("SDRT_XFIDENT").as_deref() != Ok("0");
        if dihedral && bilinear && ident_mode && manual.is_none() {
            let (tw, th) = if m[0] == 0.0 { (h, w) } else { (w, h) };
            let mut cur = self.img(k).clone();
            // per axis: a larger source is halved along that axis (pair averages, as its mip
            // levels are stored), a smaller one doubled — feather_generator node 1, a 16×256 FX-Map
            // in a 128² node: x interpolated at the source's 16 px, y averaged (sampling the 8×128
            // mip level moved a one-code step by a pixel). SDRT_XFAXES=0: both axes by mip level.
            let per_axis = crate::envc::var("SDRT_XFAXES").as_deref() != Ok("0");
            if (cur.w <= tw && cur.h <= th) || per_axis {
                // the source's format (redbrickbuilding 1008, 16-bit 32² → 128²: one code off with
                // float weights); SDRT_XFUPF=0 for float weights
                let f = if crate::envc::var("SDRT_XFUPF").as_deref() == Ok("0") { 3 } else { self.p.nodes[k].format() };
                // the halved levels are stored in the transformation's own format (stone_flat_wall 3,
                // 4096² bitmap → 128² 8-bit: 0.0078 → 0.0039, banana_skin_02 265 exact; lizard_scales 3
                // stays exact). SDRT_XFMIPF=src: the source's format (bitmaps float), 16, 8
                let mf = match crate::envc::var("SDRT_XFMIPF").as_deref() {
                    Ok("src") => if self.p.nodes[k].op & 0xFE == 0x20 { 3 } else { self.p.nodes[k].format() },
                    Ok("16") => 1,
                    Ok("8") => 0,
                    _ => n.format(),
                };
                let mut sample_fmt = self.p.nodes[k].format();
                while cur.w > tw || cur.h > th {
                    cur = cur.downsample_axes_sampled(cur.w > tw, cur.h > th, mf, sample_fmt);
                    sample_fmt = mf;
                }
                while cur.w < tw || cur.h < th {
                    cur = cur.upsample_axes_fmt(cur.w < tw, cur.h < th, wrap, f);
                }
                if identity {
                    cur.color = n.color;
                    return cur;
                }
                let mut out = Image::new(w, h, n.color, [0.0; 4]);
                for y in 0..h {
                    for x in 0..w {
                        let (sx, sy) = if m[0] != 0.0 {
                            (if m[0] > 0.0 { x } else { w - 1 - x }, if m[3] > 0.0 { y } else { h - 1 - y })
                        } else {
                            (if m[2] > 0.0 { y } else { h - 1 - y }, if m[1] > 0.0 { x } else { w - 1 - x })
                        };
                        out.px[y * w + x] = cur.at(sx, sy);
                    }
                }
                return out;
            }
        }
        // row norms |grad tx|, |grad ty| in output pixel steps (adras_atlas4 135, xf_all_probe/xf_ns:
        // 11.3, 14.1 vk and 14.1 sse2 all select level 2 on [2.20,0.69,-2.10,0.72] and level 1 on M^T)
        let du = ((m[0] * sw as f32 / w as f32).powi(2) + (m[2] * sw as f32 / h as f32).powi(2)).sqrt();
        let dv = ((m[1] * sh as f32 / w as f32).powi(2) + (m[3] * sh as f32 / h as f32).powi(2)).sqrt();
        let lod_raw = du.max(dv).log2();
        // measured: the nearest mip level, no blending between levels (lab/py/lodtest.py, xf_all_probe,
        // probe_hi: 14.1 vk and sse2 switch 0→1 and 1→2 in [k+0.49950, k+0.49960], so bias +1/2048 puts
        // threshold at k+0.4995117: led_stripe 908 lod 0.4999862 and worn_rope 298 lod 0.49999985 take
        // level 1 while geometric_flowers_v2 6383 lod 1.4990531 stays level 1)
        let mip_bias = crate::envc::var("SDRT_XFMIPBIAS").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0 / 2048.0);
        let level = match manual {
            Some(l) => l.max(0.0).round() as usize,
            None => (lod_raw.max(0.0) + 0.5 + mip_bias).floor() as usize,
        };
        // transformation builds mip levels >= 1 in its own format (white-sand 1521/1526, xfmipfmt2/3:
        // UNORM8 -> 32F xf gives float mips, 32F -> UNORM8 xf gives 8-bit mips; embedded bitmaps
        // keep float mips on non-identity, lizard_scales 3). SDRT_XFMIPF=src: source's format, =own: n.format() even for bitmaps
        let xf_mip_fmt = match crate::envc::var("SDRT_XFMIPF").as_deref() {
            Ok("src") => if self.p.nodes[k].op & 0xFE == 0x20 { 3 } else { self.p.nodes[k].format() },
            Ok("own") => n.format(),
            _ if self.p.nodes[k].op & 0xFE == 0x20 => 3,
            _ => n.format(),
        };
        // nearest filtering samples the chosen mip level too (scratch/pp2/xf.py)
        let levels = if level > 0 { Some(self.mip_levels_fmt(k, xf_mip_fmt)) } else { None };
        // never past the output's size: beyond it bilinear magnification does the rest
        let max_ups = ((w.max(h) as f32) / (sw.max(sh) as f32)).log2().floor().max(0.0) as i32;
        let ups = ((-lod_raw).round() as i32).min(max_ups)
            - crate::envc::var("SDRT_XFUPSHIFT").ok().and_then(|s| s.parse::<i32>().ok()).unwrap_or(0);
        // a non-identity matrix samples level 0 directly when magnifying (xfmag.py: offsets show
        // one bilinear step; the 2× upsample chain belongs to the identity resize above)
        let up = if bilinear && manual.is_none() && ups >= 1 && crate::envc::var("SDRT_XFUPS").is_ok() {
            // magnification: successive 2× bilinear upsamples, as many as the rounded level
            // (measured: exact for powers of 2)
            let mut up = self.img(k).upsample2(wrap);
            for _ in 1..ups {
                up = up.upsample2(wrap);
            }
            Some(up)
        } else {
            None
        };
        let src = self.img(k);
        let img: &Image = match (&levels, &up) {
            // a 1 px source has no levels: it is its own smallest level
            (Some(l), _) if !l.is_empty() => &l[(level - 1).min(l.len() - 1)],
            (_, Some(u)) => u,
            _ => src,
        };
        use rayon::prelude::*;
        let color = n.color;
        // SDRT_XFBORDER: 0 matte border (default), 1 transparent black, 2 clamp to edge, lerp weights (old)
        // the format the sampled texels are stored in (an embedded bitmap's mip levels are float)
        let src_fmt = if crate::envc::var("SDRT_XFUNORM").as_deref() == Ok("0") { 3 } else if level > 0 { xf_mip_fmt } else { self.p.nodes[k].format() };
        let border_mode: u32 = crate::envc::var("SDRT_XFBORDER").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        let edge_coverage = crate::envc::var("SDRT_XFEDGE").as_deref() != Ok("legacy");
        if crate::envc::var("SDRT_XFDBG").is_ok() {
            eprintln!("xf: out {}x{} src {}x{} img {}x{} level {} lod {} wrap {:?} matte {:?} m {:?} off {:?} bil {}", w, h, sw, sh, img.w, img.h, level, lod_raw, wrap, matte, m, off, bilinear);
        }
        let rows: Vec<Vec<[f32; 4]>> = (0..h)
            .into_par_iter()
            .map(|y| {
                (0..w)
                    .map(|x| {
                        let dx = (x as f32 + 0.5) / w as f32 - 0.5;
                        let dy = (y as f32 + 0.5) / h as f32 - 0.5;
                        let u = m[0] * dx + m[2] * dy + 0.5 + off[0];
                        let v = m[1] * dx + m[3] * dy + 0.5 + off[1];
                        let outside = (!wrap.0 && !(0.0..=1.0).contains(&u)) || (!wrap.1 && !(0.0..=1.0).contains(&v));
                        // an identity matrix with a manual level clamps to the edge texel
                        // (xfborder.py "identity manual1"; miptest.py levels 2, 3, 5)
                        let border = match border_mode {
                            _ if identity && ident_mode => None,
                            1 => Some([0.0; 4]),
                            _ => Some(if color { [matte[0], matte[1], matte[2], matte[3]] } else { Image::gray(matte[0]) }),
                        };
                        // bilinear samples past a non-tiled edge keep blending with the border
                        // (xfmag.py o1: u = 1.011 still weighs the last texel 0.32)
                        // tiled axes too: the coupled 1/256 cross weight (xfcross.py, random source)
                        if bilinear && border_mode != 2 {
                            // 14.1 blends the edge-clamped texture with the matte using float
                            // coverage, separately from the texture unit's 1/256 weights.
                            // Measured by strict_xf/probe/edges.sbsar (all tilings, mip 0/2,
                            // UNORM/float, nonzero matte). The interior sampler is unchanged.
                            if edge_coverage {
                                if let Some(matte) = border {
                                    let (tx, ty) = (u * img.w as f32 - 0.5, v * img.h as f32 - 0.5);
                                    let cx = if wrap.0 { 1.0 } else { (tx + 1.0).min(img.w as f32 - tx).clamp(0.0, 1.0) };
                                    let cy = if wrap.1 { 1.0 } else { (ty + 1.0).min(img.h as f32 - ty).clamp(0.0, 1.0) };
                                    let coverage = cx * cy;
                                    if coverage == 0.0 { return matte; }
                                    if coverage < 1.0 {
                                        let p = img.bilinear_or(u, v, wrap, None, src_fmt);
                                        return std::array::from_fn(|c| p[c] * coverage + matte[c] * (1.0 - coverage));
                                    }
                                }
                            }
                            img.bilinear_or(u, v, wrap, border, src_fmt)
                        } else if outside {
                            if color { [matte[0], matte[1], matte[2], matte[3]] } else { Image::gray(matte[0]) }
                        } else if bilinear {
                            img.bilinear(u, v, wrap)
                        } else {
                            img.nearest(u, v, wrap)
                        }
                    })
                    .collect()
            })
            .collect();
        Image { w, h, color, px: rows.into_iter().flatten().collect() }
    }

    /// Normal from a height map. Measured with an impulse (lab/py/normalimpulse.py): backward
    /// differences, n ∝ (−k (h(x) − h(x−1)), −k (h(y) − h(y−1)), 1), k = intensity × size / 256;
    /// edges clamp when not tiled and wrap when tiled. Alpha is the height.
    /// TODO: pixelsize, input2alpha off.
    fn normal(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        // a larger input is read at its own size: differences one input texel apart, bilinear at
        // the node's pixel centres, scaled by the input's size (lab/py/normalbig.py: 64² normal of a
        // 128² height, exact; as the warp's gradient). SDRT_WARPBIG=0: resized first (old)
        let big = crate::env_once!("SDRT_WARPBIG").as_deref() != Ok("0");
        let src = if big { self.images[n.inputs[0]].clone().expect("input computed") } else { self.input_at(n.inputs[0], w, h) };
        let sfmt = self.tex_format(n.inputs[0]);
        let intensity = self.param(n, "intensity", seed).map_or(1.0, |v| v[0]);
        // inversedy: a bool slot — 1 true (no word), 2 a function such as `normal_format == 1`
        // (orange-hairs node 156)
        let inv = match n.param("inversedy") {
            Slot::Const(w) => w.first().is_some_and(|&x| x != 0),
            Slot::Fn(_) => self.param(n, "inversedy", seed).is_some_and(|v| v[0] != 0.0),
            _ => false,
        };
        // output alpha: the height, always before format version 5; from 5 on only when the
        // input2alpha bool is true (slot 1, or a function) — lab/py/normalalpha2.py
        let height_alpha = self.p.version < 5
            || match n.param("input2alpha") {
                Slot::Const(_) => true,
                Slot::Fn(_) => self.param(n, "input2alpha", seed).is_some_and(|v| v[0] != 0.0),
                _ => false,
            };
        let wrap = (n.b3 & 1 != 0, n.b3 & 2 != 0);
        let ps = self.pixelsize(n, seed);
        let (sw, sh) = (src.w, src.h);
        let same = sw == w && sh == h;
        let (kx, ky) = (intensity * sw as f32 / 256.0 * ps.0, intensity * sh as f32 / 256.0 * ps.1);
        let mut o = Image::new(w, h, true, [0.0; 4]);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let (c, dx, dy) = if same {
                    let c = src.fetch(x, y, wrap)[0];
                    (c, c - src.fetch(x - 1, y, wrap)[0], c - src.fetch(x, y - 1, wrap)[0])
                } else {
                    let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                    let s = |u: f32, v: f32| src.bilinear_or(u, v, wrap, None, sfmt)[0];
                    let c = s(u, v);
                    (c, c - s(u - 1.0 / sw as f32, v), c - s(u, v - 1.0 / sh as f32))
                };
                let (vx, vy) = (-dx * kx, -dy * ky * if inv { -1.0 } else { 1.0 });
                let l = (vx * vx + vy * vy + 1.0).sqrt();
                o.px[y as usize * w + x as usize] = [vx / l * 0.5 + 0.5, vy / l * 0.5 + 0.5, 1.0 / l * 0.5 + 0.5, if height_alpha { c } else { 1.0 }];
            }
        }
        o
    }

    /// Blur: a separable box; each tap weighs its pixel's overlap with [−r, r], over 2r, with
    /// r = intensity × size / 256 (measured with impulses, lab/py/blurimpulse2.py).
    /// TODO: edges when not tiled (clamp assumed), pixelsize.
    fn blur(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let inten = self.param(n, "intensity", seed).map_or(10.0, |v| v[0]);
        let wrap = (n.b3 & 1 != 0, n.b3 & 2 != 0);
        let ps = self.pixelsize(n, seed);
        // a larger input: the first (horizontal) pass fetches it at its own size — each fetch (a
        // single tap, or a pair's bilinear fetch at its fractional offset) at node-pixel offsets
        // from the node's centres; the second pass reads the node-size intermediate as usual
        // (lab/py/bigblur.py). SDRT_BLURBIG=0: the input resized first (old)
        let nat = self.images[n.inputs[0]].clone().expect("input computed");
        if (nat.w != w || nat.h != h) && nat.w >= w && nat.h >= h && crate::env_once!("SDRT_BLURBIG").as_deref() != Ok("0") {
            let rx = inten * w as f32 / 256.0 * ps.0;
            let ry = inten * h as f32 / 256.0 * ps.1;
            let fmt = self.tex_format(n.inputs[0]);
            // SDRT_BLURYX=1 (test): the vertical pass first
            let yx = crate::env_once!("SDRT_BLURYX").is_ok();
            let (r1, along_x) = if yx { (ry, false) } else { (rx, true) };
            let (fe, fo) = (blur_fetches(r1, 0), blur_fetches(r1, 1));
            let mut tmp = Image::new(w, h, nat.color, [0.0; 4]);
            for y in 0..h {
                for x in 0..w {
                    let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                    let mut acc = [0.0f32; 4];
                    let par = if along_x { x } else { y };
                    for &(off, wt) in if par % 2 == 0 { &fe } else { &fo } {
                        let p = if along_x { nat.bilinear_or(u + off / w as f32, v, wrap, None, fmt) } else { nat.bilinear_or(u, v + off / h as f32, wrap, None, fmt) };
                        for c in 0..4 { acc[c] += p[c] * wt; }
                    }
                    tmp.px[y * w + x] = acc;
                }
            }
            return if yx {
                box_pass_parity(&tmp, &blur_taps(rx, 0), &blur_taps(rx, 1), (1, 0), wrap)
            } else {
                box_pass_parity(&tmp, &blur_taps(ry, 0), &blur_taps(ry, 1), (0, 1), wrap)
            };
        }
        let src = self.input_at(n.inputs[0], w, h);
        let rx = inten * w as f32 / 256.0 * ps.0;
        let ry = inten * h as f32 / 256.0 * ps.1;
        let tmp = box_pass_parity(&src, &blur_taps(rx, 0), &blur_taps(rx, 1), (1, 0), wrap);
        box_pass_parity(&tmp, &blur_taps(ry, 0), &blur_taps(ry, 1), (0, 1), wrap)
    }

    /// Warp: samples input1 at uv + intensity·(size/256)·(backward differences of the gradient
    /// input), measured (lab/py/warpfit.py). TODO: v9's leading word, pixelsize.
    /// The texture format a sampled input is read in: its node's storage format (UNORM ≤ 1
    /// interpolates in 16-bit integers: the texture unit), or float (3) with SDRT_TEXFLOAT=1.
    fn tex_format(&self, k: usize) -> u32 {
        if crate::envc::var("SDRT_TEXFLOAT").is_ok() { 3 } else { self.p.nodes[k].format() }
    }

    fn warp(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let inten = self.param(n, "intensity", seed).map_or(1.0, |v| v[0]);
        let wrap = (n.b3 & 1 != 0, n.b3 & 2 != 0);
        let ps = self.pixelsize(n, seed);
        let fmt = self.tex_format(n.inputs[0]);
        let gfmt = self.tex_format(n.inputs[1]);
        // Inputs are read at their own size (the cooker only inserts an upscale for smaller ones): the
        // gradient's differences are one *gradient* texel apart, bilinear at the node's pixel centres,
        // scaled by the gradient's size (lab/py/warpbig.py: 64² warp, 128² gradient — exact, scale
        // 127.5 ≈ 128; the node's size and one node pixel gave 0.003). Same size: backward differences
        // of the texels, as before. The source is sampled at its own size too. SDRT_WARPBIG=0: both
        // resized to the node first (old).
        let big = crate::env_once!("SDRT_WARPBIG").as_deref() != Ok("0");
        let (src, grad) = if big {
            (self.images[n.inputs[0]].clone().expect("input computed"), self.images[n.inputs[1]].clone().expect("input computed"))
        } else {
            (self.input_at(n.inputs[0], w, h), self.input_at(n.inputs[1], w, h))
        };
        let (gw, gh) = (grad.w, grad.h);
        let (kx, ky) = (inten * gw as f32 / 256.0 * ps.0, inten * gh as f32 / 256.0 * ps.1);
        let gv = |p: [f32; 4]| -> f32 { if grad.color { p[3] } else { p[0] } };
        let same = gw == w && gh == h;
        let mut o = Image::new(w, h, n.color, [0.0; 4]);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let (u0, v0) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                let (du, dv) = if same {
                    let c = gv(grad.fetch(x, y, wrap));
                    (kx * (c - gv(grad.fetch(x - 1, y, wrap))), ky * (c - gv(grad.fetch(x, y - 1, wrap))))
                } else {
                    let gs = |u: f32, v: f32| gv(grad.bilinear_or(u, v, wrap, None, gfmt));
                    let c = gs(u0, v0);
                    (kx * (c - gs(u0 - 1.0 / gw as f32, v0)), ky * (c - gs(u0, v0 - 1.0 / gh as f32)))
                };
                o.px[y as usize * w + x as usize] = src.bilinear_or(u0 + du, v0 + dv, wrap, None, fmt);
            }
        }
        o
    }

    /// Directional warp: samples input1 at uv + intensity·h/256·(cos, sin) of the angle (turns),
    /// h = the intensity input (measured, lab/py/warpfit.py).
    fn directional_warp(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        // a larger source is sampled at its own size (as the warp's); SDRT_WARPBIG=0: resized (old)
        let src = if crate::env_once!("SDRT_WARPBIG").as_deref() != Ok("0") { self.images[n.inputs[0]].clone().expect("input computed") } else { self.input_at(n.inputs[0], w, h) };
        let amount = if crate::env_once!("SDRT_WARPBIG").as_deref() == Ok("0") {
            self.input_at(n.inputs[1], w, h)
        } else {
            self.images[n.inputs[1]].clone().expect("input computed")
        };
        let afmt = self.tex_format(n.inputs[1]);
        let same = (amount.w, amount.h) == (w, h);
        // the default intensity is 10 (raw_copper 94, no intensity word: frozen 0.50 → 0.0007)
        let dflt: f32 = crate::envc::var("SDRT_DWDEF").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0);
        let inten = self.param(n, "intensity", seed).map_or(dflt, |v| v[0]);
        let ang = self.param(n, "angle", seed).map_or(0.0, |v| v[0]) * std::f32::consts::TAU;
        let wrap = (n.b3 & 1 != 0, n.b3 & 2 != 0);
        let ps = self.pixelsize(n, seed);
        if crate::envc::var("SDRT_DWDBG").is_ok() { eprintln!("dirwarp {} inten {} ang {} ps {:?}", n.index, inten, ang, ps); }
        // v9's third parameter (flags bit 7): an offset subtracted from the intensity input, so the
        // warp goes both ways (marble 21: 0.92, mean 0.31 → exact; as a multiplier or an angle:
        // no). Absent (older files): 0. SDRT_DW3=none / mul / ang: test variants
        let third = self.param(n, "v9third", seed).map(|v| v[0]);
        let dw3 = crate::env_once!("SDRT_DW3").unwrap_or("sub");
        let (inten, ang, sub) = match (third, dw3) {
            (Some(t), "mul") => (inten * t, ang, 0.0),
            (Some(t), "sub") => (inten, ang, t),
            (Some(t), "ang") => (inten, ang + t * std::f32::consts::TAU, 0.0),
            _ => (inten, ang, 0.0),
        };
        let (dx, dy) = (ang.cos() * inten / 256.0 * ps.0, ang.sin() * inten / 256.0 * ps.1);
        let fmt = self.tex_format(n.inputs[0]);
        let mut o = Image::new(w, h, n.color, [0.0; 4]);
        use rayon::prelude::*;
        o.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let u0 = (x as f32 + 0.5) / w as f32;
                let v0 = (y as f32 + 0.5) / h as f32;
                let a = if same {
                    amount.px[y * w + x]
                } else {
                    amount.bilinear_or(u0, v0, wrap, None, afmt)
                };
                let m = if amount.color { a[3] } else { a[0] } - sub;
                let u = u0 + dx * m;
                let v = v0 + dy * m;
                row[x] = src.bilinear_or(u, v, wrap, None, fmt);
            }
        });
        o
    }

    /// Curve: cubic Bézier segments between points (position, left handle, right handle); gray
    /// has one curve, color five (luminance applied to R, G, B first, then R, G, B, A); an empty
    /// curve is the identity. Measured (lab/py/curvetest.py).
    fn curve(&mut self, n: &Node) -> Image {
        let (w, h) = n.size();
        let src = self.input_at(n.inputs[0], w, h);
        let curves = self.curve_points(n);
        let mut out = Image::new(w, h, n.color, [0.0; 4]);
        use rayon::prelude::*;
        let color = n.color;
        // byte3 bit 1 (seen in v8 files): the curve is evaluated along x — (x + ½)/w — and the input
        // is not read (orgu-kablo 151, 14.1 vk: a horizontal profile with a zero input). SDRT_CURVEX=0: off
        let along_x = n.b3 & 2 != 0 && crate::envc::var("SDRT_CURVEX").as_deref() != Ok("0");
        let conv: u32 = crate::envc::var("SDRT_CURVEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        // the reference evaluates a curve through a table of 1025 values at x = i/1024, linearly
        // interpolated with the fraction in 1/256 steps (the texture unit's) — chinese_tile_ 45: a
        // point at (0.001, 0.114), input 42/65535 → 0.07481 (exact Bézier: 0.10809); 8 pixels 0.033
        // → exact, fewest 2-code pixels with 1/256 fractions. SDRT_CURVELUT=0: exact Bézier, =float:
        // unrounded fractions; SDRT_CURVELUTN: table size (test)
        let lut_mode = crate::envc::var("SDRT_CURVELUT").ok();
        let lut_n: usize = crate::envc::var("SDRT_CURVELUTN").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
        let luts: Option<Vec<Vec<f32>>> = if lut_mode.as_deref() == Some("0") { None } else {
            // SDRT_CURVELUTQ=16 / half: table values stored as UNORM16 / f16 (test)
            let q = crate::envc::var("SDRT_CURVELUTQ").ok();
            let qf = |v: f32| -> f32 { match q.as_deref() { Some("16") => (v.clamp(0.0, 1.0) * 65535.0).round() / 65535.0, Some("half") => crate::image::half_trunc(v), _ => v } };
            Some(curves.iter().map(|c| (0..=lut_n).map(|i| qf(bezier_curve(c, i as f32 / lut_n as f32))).collect()).collect())
        };
        let frac256 = lut_mode.as_deref() != Some("float");
        let eval = |ci: usize, x: f32| -> f32 {
            match &luts {
                None => bezier_curve(&curves[ci], x),
                Some(l) if curves[ci].is_empty() => { let _ = l; x }
                Some(l) => {
                    let t = x.clamp(0.0, 1.0) * lut_n as f32;
                    let i = (t.floor() as usize).min(lut_n - 1);
                    let mut f = t - i as f32;
                    if frac256 { f = (f * 256.0).round() / 256.0; }
                    l[ci][i] + (l[ci][i + 1] - l[ci][i]) * f
                }
            }
        };
        out.px.par_iter_mut().enumerate().for_each(|(k, px)| {
            let s = if along_x {
                let x = (k % w) as f32;
                let t = match conv { 1 => x / w as f32, 2 => x / (w as f32 - 1.0), _ => (x + 0.5) / w as f32 };
                [t, t, t, t]
            } else { src.px[k] };
            *px = if color {
                let mut c = s;
                for ch in 0..3 {
                    c[ch] = eval(0, c[ch]);
                }
                for ch in 0..4 {
                    c[ch] = eval(ch + 1, c[ch]);
                }
                c
            } else {
                Image::gray(eval(0, s[0]))
            };
        });
        out
    }

    /// A curve node's Bézier curves: one (gray) or five (colour: luminance, R, G, B, A), each a
    /// u32 count then its points (x, y, left handle, right handle), shared by the CPU path and
    /// try_gpu_curve.
    fn curve_points(&self, n: &Node) -> Vec<Vec<[f32; 6]>> {
        let p = self.p;
        let mut curves: Vec<Vec<[f32; 6]>> = vec![];
        let mut o = n.data;
        // each curve: u32 n, n points (own length; all 52 color curves of the Adobe corpus, e.g.
        // brushed-gold node 77 with empty R, G, B, A curves after a 4-point luminance curve)
        for _ in 0..(if n.color { 5 } else { 1 }) {
            let k = (p.w(o) as usize).min(n.count as usize);
            let pts = (0..k).map(|i| std::array::from_fn(|j| p.f(o + 4 + 24 * i + 4 * j))).collect();
            curves.push(pts);
            o += 4 + 24 * k;
        }
        curves
    }

    /// Distance: 1 − d/D, d = Euclidean distance in pixels to the nearest mask pixel (> 0.5), D =
    /// distance × size/256 (measured, lab/py/distprobe.py, distsrc.py). TODO: the mode bits, color
    /// alpha with combine.
    fn distance(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let mask = self.input_at(n.inputs[0], w, h);
        let src = n.inputs.get(1).map(|&k| self.input_at(k, w, h));
        // pixelsize weighs the axes: a pixel step counts 1/ps (isotropic ps = a radius × ps)
        let dist = self.param(n, "distance", seed).map_or(10.0, |v| v[0]) * w as f32 / 256.0;
        let ps = self.pixelsize(n, seed);
        let wrap_x = n.b3 & 1 != 0;
        let wrap_y = n.b3 & 2 != 0;
        let inside: Vec<bool> = mask.px.iter().map(|p| (if mask.color { p[3] } else { p[0] }) > 0.5).collect();
        let quant = crate::envc::var("SDRT_DISTQ").as_deref() != Ok("0");
        // sbsrender 14.1's distance (Codex, cx/distance14): exact query positions, UNORM16 seeds,
        // a fused metric, metric modes (Euclidean / Manhattan / Chebyshev). 11.3 differs (NOTES
        // "Version difference"); default: 14.1 for every file version (the target engine, user
        // decision 2026-11-01). SDRT_DIST14=0 gives 11.3's distance.
        let v14 = crate::envc::var("SDRT_DIST14").as_deref() != Ok("0");
        let mode14 = self.param(n, "mode", seed).map_or((n.flags >> 4) & 3, |v| v[0] as u32);
        let code = |v: usize, n: usize| -> f32 { ((v as f64 + 0.5) / n as f64 * 65535.0).round() as f32 };
        let xc: Vec<f32> = (0..w).map(|x| code(x, w)).collect();
        let yc: Vec<f32> = (0..h).map(|y| code(y, h)).collect();
        let dmax = self.param(n, "distance", seed).map_or(10.0, |v| v[0]) / 256.0;
        let inv = 1.0f32 / dmax;
        let delta14 = |p: usize, s: usize, n: usize, wrap: bool| -> f32 {
            let mut d = (p as f32 + 0.5) / n as f32 - code(s, n) / 65535.0;
            if wrap && d.abs() > 0.5 { d -= d.signum(); }
            d
        };
        let qdelta = |k: usize, s: usize| -> (f32, f32) {
            if v14 { return (delta14(k % w, s % w, w, wrap_x) * (1.0 / ps.0),
                              delta14(k / w, s / w, h, wrap_y) * (1.0 / ps.1)); }
            let delta = |a: f32, b: f32, wrap: bool| -> f32 {
                let mut d = a - b;
                if wrap && d.abs() > 32767.0 { d -= d.signum() * 65535.0; }
                d / 65535.0
            };
            let du = delta(xc[k % w], xc[s % w], wrap_x) / ps.0;
            let dv = delta(yc[k / w], yc[s / w], wrap_y) / ps.1;
            (du, dv)
        };
        // Selection contracts the Y square and addition; final falloff keeps the
        // existing unfused dot product (disttie: 12/12 bit-exact).
        let qd2 = |k: usize, s: usize| -> f32 {
            if v14 {
                let du = delta14(k % w, s % w, w, wrap_x) * (1.0 / ps.0);
                let dv = delta14(k / w, s / w, h, wrap_y) * (1.0 / ps.1);
                return match mode14 { 0 => du.mul_add(du, dv * dv), 1 => du.abs() + dv.abs(), _ => du.abs().max(dv.abs()) };
            }
            let delta = |a: f32, b: f32, wrap: bool| -> f32 {
                let mut d = a / 65535.0 - b / 65535.0;
                if wrap && d.abs() > 0.5 { d -= d.signum(); }
                d
            };
            let du = delta(xc[k % w], xc[s % w], wrap_x) * (1.0 / ps.0);
            let dv = delta(yc[k / w], yc[s / w], wrap_y) * (1.0 / ps.1);
            dv.mul_add(dv, du * du)
        };
        let qd = |k: usize, s: usize| -> f32 {
            let (du, dv) = qdelta(k, s);
            if v14 {
                match mode14 { 0 => du.mul_add(du, dv * dv).sqrt(), 1 => du.abs() + dv.abs(), _ => du.abs().max(dv.abs()) }
            } else { (du * du + dv * dv).sqrt() }
        };
        // Jump flooding. Strict improvement retains the first candidate at a tie.
        // Neighbour reads repeat on tiled axes and clamp on untiled axes.
        let mut near: Vec<usize> = inside.iter().enumerate().map(|(k, &s)| if s { k } else { usize::MAX }).collect();
        let dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)];
        let mut step = w.max(h) / 2;
        while step > 0 {
            let mut next = near.clone();
            for y in 0..h {
                for x in 0..w {
                    let k = y * w + x;
                    let mut best = if near[k] == usize::MAX { f32::INFINITY } else { qd2(k, near[k]) };
                    for &(dx, dy) in &dirs {
                        let at = |p: usize, d: i32, n: usize, wrap: bool| -> usize {
                            // Jumps are in normalized coordinates, from the quantized current
                            // position. Half-texel jumps on the shorter axis choose opposite
                            // sides in the two image halves (varied probes).
                            let origin = if v14 { (p as f32 + 0.5) / n as f32 } else { code(p, n) / 65535.0 };
                            let uv = origin + d as f32 * step as f32 / w.max(h) as f32;
                            let v = (uv * n as f32).floor() as i64;
                            if wrap { v.rem_euclid(n as i64) as usize } else { v.clamp(0, n as i64 - 1) as usize }
                        };
                        let q = near[at(y, dy, h, wrap_y) * w + at(x, dx, w, wrap_x)];
                        if q == usize::MAX { continue; }
                        let d = qd2(k, q);
                        if d < best { best = d; next[k] = q; }
                    }
                }
            }
            near = next;
            step /= 2;
        }
        let (d2, near) = if quant {
            (vec![0.0; w * h], near)
        } else {
            edt(&inside, w, h, wrap_x && wrap_y, 1.0 / ps.0, 1.0 / ps.1)
        };
        let mut o = Image::new(w, h, n.color, [0.0; 4]);
        for k in 0..w * h {
            let q = quant && near[k] != usize::MAX;
            if near[k] != usize::MAX && crate::envc::var("SDRT_DISTDBG").ok().and_then(|v| v.parse::<usize>().ok()) == Some(k) {
                eprintln!("dist px {} ({},{}) near {} ({},{}) d2 {} qd {} dmax {} ps {:?}", k, k % w, k / w, near[k], near[k] % w, near[k] / w, d2[k], qd(k, near[k]), dmax, ps);
            }
            let (within, v) = if near[k] == usize::MAX {
                (false, 0.0)
            } else if q {
                let d = if v14 && inside[k] { 0.0 } else { qd(k, near[k]) };
                (d < dmax, (-d).mul_add(inv, 1.0).max(0.0))
            } else {
                let d = d2[k].sqrt();
                (d < dist, (1.0 - d / dist).max(0.0))
            };
            o.px[k] = match &src {
                // with a source (measured, lab/py/distsrc.py): combine → source at the nearest mask
                // pixel × (1 − d/D); otherwise that source within D, the source here beyond
                Some(s) => {
                    let c = if n.flags & 2 != 0 {
                        let p = if near[k] == usize::MAX { s.px[k] } else { s.px[near[k]] };
                        // color: the falloff goes to alpha only; rgb as without combine
                        let q = if within { p } else { s.px[k] };
                        if n.color { [q[0], q[1], q[2], p[3] * v] } else { p.map(|x| x * v) }
                    } else if within {
                        s.px[near[k]]
                    } else {
                        s.px[k]
                    };
                    if n.color { c } else { Image::gray(c[0]) }
                }
                _ => if n.color { [v, v, v, 1.0] } else { Image::gray(v) },
            };
        }
        o
    }

    /// Emboss: s = intensity·(size/256)·(bx cos 2πa + by sin 2πa) with backward differences of the
    /// height; rgb += s·highlight where s > 0, s·(1 − shadow) where s < 0 (measured,
    /// lab/py/embossprobe*.py). TODO: pixelsize, alpha.
    fn emboss(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let src = self.input_at(n.inputs[0], w, h);
        // a larger height is read at its own size (as the normal's: input-texel differences,
        // bilinear at the node's centres, scaled by its size); SDRT_WARPBIG=0: resized (old)
        let big = crate::env_once!("SDRT_WARPBIG").as_deref() != Ok("0");
        let hgt = if big { self.images[n.inputs[1]].clone().expect("input computed") } else { self.input_at(n.inputs[1], w, h) };
        let hfmt = self.tex_format(n.inputs[1]);
        let inten = self.param(n, "intensity", seed).map_or(5.0, |v| v[0]);
        let ang = self.param(n, "lightangle", seed).map_or(0.0, |v| v[0]) * std::f32::consts::TAU;
        let hl = self.param(n, "highlight", seed).unwrap_or(vec![1.0; 4]);
        let sh = self.param(n, "shadow", seed).unwrap_or(vec![0.0, 0.0, 0.0, 1.0]);
        let wrap = (n.b3 & 1 != 0, n.b3 & 2 != 0);
        let ps = self.pixelsize(n, seed);
        let (hw, hh) = (hgt.w, hgt.h);
        let same = hw == w && hh == h;
        let (kx, ky) = (inten * hw as f32 / 256.0 * ps.0, inten * hh as f32 / 256.0 * ps.1);
        let (ca, sa) = (ang.cos(), ang.sin());
        let hv = |p: [f32; 4]| -> f32 { if hgt.color { p[3] } else { p[0] } };
        let g = |x: i64, y: i64| -> f32 { hv(hgt.fetch(x, y, wrap)) };
        let mut o = Image::new(w, h, n.color, [0.0; 4]);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let (gx, gy) = if same {
                    let c = g(x, y);
                    (c - g(x - 1, y), c - g(x, y - 1))
                } else {
                    let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                    let b = |u: f32, v: f32| hv(hgt.bilinear_or(u, v, wrap, None, hfmt));
                    let c = b(u, v);
                    (c - b(u - 1.0 / hw as f32, v), c - b(u, v - 1.0 / hh as f32))
                };
                let s = kx * gx * ca + ky * gy * sa;
                let p = src.px[y as usize * w + x as usize];
                let mut q = p;
                for ch in 0..3 {
                    let col = if n.color { if s > 0.0 { hl[ch] } else { 1.0 - sh[ch] } } else if s > 0.0 { hl[0] } else { 1.0 - sh[0] };
                    q[ch] = p[ch] + s * col;
                }
                o.px[y as usize * w + x as usize] = if n.color { q } else { Image::gray(q[0]) };
            }
        }
        o
    }

    /// Sharpen: x + intensity·(4x − the four neighbours) (measured with an impulse).
    fn sharpen(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        // a larger input: neighbours one input texel apart, bilinear at the node's centres
        let big = crate::env_once!("SDRT_WARPBIG").as_deref() != Ok("0");
        let nat = self.images[n.inputs[0]].clone().expect("input computed");
        let sfmt = self.tex_format(n.inputs[0]);
        let larger = big && (nat.w != w || nat.h != h);
        let src = self.input_at(n.inputs[0], w, h);
        let inten = self.param(n, "intensity", seed).map_or(1.0, |v| v[0]);
        let wrap = (n.b3 & 1 != 0, n.b3 & 2 != 0);
        let ps = self.pixelsize(n, seed);
        let mut o = Image::new(w, h, n.color, [0.0; 4]);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let (c, l, r, u, d) = if larger {
                    let (uu, vv) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                    let (ox, oy) = (1.0 / nat.w as f32, 1.0 / nat.h as f32);
                    let b = |u: f32, v: f32| nat.bilinear_or(u, v, wrap, None, sfmt);
                    (b(uu, vv), b(uu - ox, vv), b(uu + ox, vv), b(uu, vv - oy), b(uu, vv + oy))
                } else {
                    (src.fetch(x, y, wrap), src.fetch(x - 1, y, wrap), src.fetch(x + 1, y, wrap), src.fetch(x, y - 1, wrap), src.fetch(x, y + 1, wrap))
                };
                let mut q = [0.0; 4];
                for ch in 0..4 {
                    // pixelsize weighs each axis' second difference (lab/py/pixsizeprobe3.py)
                    q[ch] = c[ch] + inten * ((2.0 * c[ch] - l[ch] - r[ch]) * ps.0 + (2.0 * c[ch] - u[ch] - d[ch]) * ps.1);
                }
                o.px[y as usize * w + x as usize] = if n.color { q } else { Image::gray(q[0]) };
            }
        }
        o
    }

    /// Gradient (dynamic): input2 sampled bilinearly at (input1, coordinate), or (coordinate,
    /// input1) when vertical (byte3 bit 3); both mapped to texel centres, t → (t (n−1) + ½) / n
    /// (measured, lab/py/dynfit.py). TODO: addressing repeat (wraps t here).
    fn dyn_gradient(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let src = self.input_at(n.inputs[0], w, h);
        let grad = self.img(n.inputs[1]).clone();
        let gfmt = self.tex_format(n.inputs[1]);
        let coord = self.param(n, "coordinate", seed).map_or(0.0, |v| v[0]);
        let vertical = n.b3 & 8 != 0;
        let repeat = n.b3 & 1 != 0;
        let mut o = Image::new(w, h, n.color, [0.0; 4]);
        for k in 0..w * h {
            let p = src.px[k];
            let t = if src.color { p[3] } else { p[0] };
            let t = if repeat { t.rem_euclid(1.0) } else { t.clamp(0.0, 1.0) };
            let (a, b) = if vertical { (coord, t) } else { (t, coord) };
            let u = (a * (grad.w as f32 - 1.0) + 0.5) / grad.w as f32;
            let v = (b * (grad.h as f32 - 1.0) + 0.5) / grad.h as f32;
            let c = if crate::envc::var("SDRT_TEXFLOAT").is_ok() { grad.bilinear(u, v, (repeat, repeat)) } else { grad.bilinear_or(u, v, (repeat, repeat), None, gfmt) };
            o.px[k] = if n.color { c } else { Image::gray(c[0]) };
        }
        o
    }

    /// Directional motion blur.  The GPU implements an off-axis box by
    /// shearing to a dominant-axis coordinate system, applying the regular
    /// one-dimensional box, and shearing back.  Its bilinear texture units
    /// round interpolation weights (not coordinates) to 1/256.
    /// Legacy motion blur (op 0x12/0x13, Designer's old "motionblur" atomic; inputs input1, inputgradient;
    /// intensity, shiftangle): per pixel, six equal texture-unit samples at (-2.5..2.5)/6 of a segment
    /// along the backward-difference slope of the gradient input (alpha for color, R for gray; read at
    /// its own size, scaled by gw*ps.x, gh*ps.y), rotated by shiftangle*2π, with UV length
    /// intensity*|slope|*1.005/128 (= step 0.335*intensity/256). Measured exact on synthetic 1D/2D/edge
    /// probes (lab/py/legmb.py, scratch/t/lmb/lmb_edges.py) and frozen nian-lun 264.
    fn legacy_motion_blur(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let big = crate::env_once!("SDRT_WARPBIG").as_deref() != Ok("0");
        let src = if big { self.images[n.inputs[0]].clone().expect("input computed") } else { self.input_at(n.inputs[0], w, h) };
        let grad = match n.inputs.get(1) {
            Some(&k) if k < self.images.len() => if big { self.images[k].clone().expect("input computed") } else { self.input_at(k, w, h) },
            _ => return self.input_at(n.inputs[0], w, h).as_ref().clone(),
        };
        let sfmt = self.tex_format(n.inputs[0]);
        let gfmt = self.tex_format(n.inputs[1]);
        let inten = self.param(n, "intensity", seed).map_or(1.0, |v| v[0]);
        let shift = self.param(n, "shiftangle", seed).map_or(0.0, |v| v[0]) * std::f32::consts::TAU;
        let wrap = (n.b3 & 1 != 0, n.b3 & 2 != 0);
        let ps = self.pixelsize(n, seed);
        let fac: f32 = crate::envc::var("SDRT_LMBF").ok().and_then(|v| v.parse().ok()).unwrap_or(1.005);
        let (gw, gh) = (grad.w, grad.h);
        let same = gw == w && gh == h;
        let gv = |p: [f32; 4]| -> f32 { if grad.color { p[3] } else { p[0] } };
        let (cs, ss) = (shift.cos(), shift.sin());
        let scale = inten * fac / 128.0;
        let mut o = Image::new(w, h, n.color, [0.0; 4]);
        use rayon::prelude::*;
        o.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let y = y as i64;
            for x in 0..w as i64 {
                let (cu, cv) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                let (dx, dy) = if same {
                    let c = gv(grad.fetch(x, y, wrap));
                    (c - gv(grad.fetch(x - 1, y, wrap)), c - gv(grad.fetch(x, y - 1, wrap)))
                } else {
                    let gs = |u: f32, v: f32| gv(grad.bilinear_or(u, v, wrap, None, gfmt));
                    let c = gs(cu, cv);
                    (c - gs(cu - 1.0 / gw as f32, cv), c - gs(cu, cv - 1.0 / gh as f32))
                };
                let (gx, gy) = (dx * gw as f32 * ps.0, dy * gh as f32 * ps.1);
                let (ux, uy) = ((gx * cs - gy * ss) * scale, (gx * ss + gy * cs) * scale);
                let mut acc = [0.0f32; 4];
                for i in 0..6 {
                    let t = (i as f32 + 0.5) / 6.0 - 0.5;
                    let p = src.bilinear_or(cu + t * ux, cv + t * uy, wrap, None, sfmt);
                    for c in 0..4 { acc[c] += p[c]; }
                }
                let q = acc.map(|v| v / 6.0);
                row[x as usize] = if n.color { q } else { Image::gray(q[0]) };
            }
        });
        o
    }

    fn motion_blur(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let src = self.input_at(n.inputs[0], w, h);
        let inten = self.param(n, "intensity", seed).map_or(10.0, |v| v[0]);
        let ang = self.param(n, "angle", seed).map_or(0.0, |v| v[0]) * std::f32::consts::TAU;
        let wrap = (n.b3 & 1 != 0, n.b3 & 2 != 0);
        let ps = self.pixelsize(n, seed);
        let r = inten * w as f32 / 256.0;
        let (dx, dy) = (ang.cos() * ps.0, ang.sin() * ps.1);
        // Ties choose Y, as measured at exactly 45 degrees.
        let y_major = dy.abs() >= dx.abs();
        let major = if y_major { dy.abs() } else { dx.abs() };
        if major == 0.0 {
            return (*src).clone();
        }
        let axis_aligned = dx.abs().min(dy.abs()) < 1e-6;
        // On an exact axis, pixelsize scales the entire radius. The half-texel
        // shear correction applies only off-axis (strict_xf/probe/motion.sbsar).
        let radius = if axis_aligned && crate::envc::var("SDRT_DMBRADIUS").as_deref() != Ok("legacy") {
            r * major
        } else { 0.5 + (r - 0.5) * major };
        let taps = box_taps(radius);
        // SDRT_DMB: "pair" = the box pass with blur's paired bilinear fetches; "tex" = texture-unit
        // samples (coupled weights, UNORM integers); "all" = both
        let dmb = crate::envc::var("SDRT_DMB").unwrap_or_default();
        let fmt = self.tex_format(n.inputs[0]);
        let tex = |img: &Image, u: f32, v: f32| -> [f32; 4] {
            if dmb != "float" { img.bilinear_or(u, v, wrap, None, fmt) } else { img.bilinear_256(u, v, wrap) }
        };
        // SD does not build the two shear passes for the one-/three-tap
        // kernels.  It samples each major-axis tap directly instead.  This
        // also preserves the established axis-aligned path.
        if taps.len() <= 3 || axis_aligned {
            let mut o = Image::new(w, h, n.color, [0.0; 4]);
            use rayon::prelude::*;
            o.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                for x in 0..w {
                    let mut acc = [0.0f32; 4];
                    for &(t, wt) in &taps {
                        let t = t as f32;
                        let (ox, oy) = (t * dx / major, t * dy / major);
                        let u = (x as f32 + 0.5 + ox) / w as f32;
                        let v = (y as f32 + 0.5 + oy) / h as f32;
                        // Exact axes retain the historical point path.  The
                        // small off-axis kernel still goes through the GPU
                        // texture unit, whose bilinear fractions are 1/256.
                        let p = if axis_aligned { src.bilinear(u, v, wrap) }
                                else { src.bilinear_256(u, v, wrap) };
                        for c in 0..4 { acc[c] += p[c] * wt; }
                    }
                    row[x] = acc;
                }
            });
            return o;
        }
        let mut sheared = Image::new(w, h, n.color, [0.0; 4]);
        let mut o = Image::new(w, h, n.color, [0.0; 4]);
        use rayon::prelude::*;
        sheared.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let s = if y_major {
                    (y as f32 + 0.5 - h as f32 * 0.5) * dx / dy
                } else {
                    (x as f32 + 0.5 - w as f32 * 0.5) * dy / dx
                };
                let (u, v) = if y_major {
                    ((x as f32 + 0.5 + s) / w as f32, (y as f32 + 0.5) / h as f32)
                } else {
                    ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5 + s) / h as f32)
                };
                row[x] = tex(&src, u, v);
            }
        });
        let dir = if y_major { (0, 1) } else { (1, 0) };
        // SDRT_DMBSEAM=clamp: the box along the major axis clamps at the sheared image's edges
        let bwrap = match crate::envc::var("SDRT_DMBSEAM").as_deref() {
            Ok("clamp") => if y_major { (wrap.0, false) } else { (false, wrap.1) },
            _ => wrap,
        };
        // the sheared image continues past the seam (quilt node 200, tiled, 157.5°: the wrapped
        // sheared image put a 0.15 error on the seam columns; continued: 0.0014). SDRT_DMBSEAM=wrap|clamp = old
        let pad = !matches!(crate::envc::var("SDRT_DMBSEAM").as_deref(), Ok("wrap") | Ok("clamp"));
        let padw: Option<f32> = crate::envc::var("SDRT_DMBPAD").ok().and_then(|v| v.parse().ok());
        let qfmt = if crate::envc::var("SDRT_DMBQ").is_ok() { n.format() } else { 3 };
        // on an untiled minor axis the sheared image is wider than the node by the largest shear
        // (silk_from_bitmap 109: 45°, 2048², x untiled — clamped to w columns the unshear lost the
        // left edge's band below row ~1050, max 1.0; GPU and CPU engines agree). SDRT_DMBEXT=0: off
        let ext = crate::envc::var("SDRT_DMBEXT").as_deref() != Ok("0") && pad
            && if y_major { !wrap.0 } else { !wrap.1 };
        let ep = if ext {
            let smax = if y_major { h as f32 * 0.5 * (dx / dy).abs() } else { w as f32 * 0.5 * (dy / dx).abs() };
            smax.ceil() as usize + 2
        } else { 0 };
        let (bw, bh) = if y_major { (w + 2 * ep, h) } else { (w, h + 2 * ep) };
        let blurred = if pad {
            // the sheared image continued past the seam: each box tap samples the source at its own
            // unwrapped sheared position
            let mut b = Image::new(bw, bh, n.color, [0.0; 4]);
            b.px.par_chunks_mut(bw).enumerate().for_each(|(by, row)| {
                for bx in 0..bw {
                    // (x, y) in node pixels: the padding sits before the first column (row)
                    let (x, y) = if y_major { (bx as f32 - ep as f32, by as f32) } else { (bx as f32, by as f32 - ep as f32) };
                    let mut acc = [0.0f32; 4];
                    for &(t, wt) in &taps {
                        let (mut xx, mut yy) = if y_major { (x, y + t as f32) } else { (x + t as f32, y) };
                        // SDRT_DMBPAD=P: the sheared image has P columns (rows) of continuation
                        if let Some(pw) = padw {
                            let (n_, c) = if y_major { (h as f32, &mut yy) } else { (w as f32, &mut xx) };
                            if *c < -pw || *c >= n_ + pw { *c = c.rem_euclid(n_); }
                        }
                        let sh = if y_major { (yy + 0.5 - h as f32 * 0.5) * dx / dy } else { (xx + 0.5 - w as f32 * 0.5) * dy / dx };
                        let (u, v) = if y_major { ((xx + 0.5 + sh) / w as f32, (yy + 0.5) / h as f32) } else { ((xx + 0.5) / w as f32, (yy + 0.5 + sh) / h as f32) };
                        let mut p = tex(&src, u, v);
                        // the sheared image is stored in the node's format before the box reads it
                        if qfmt <= 1 { p = p.map(|v| crate::image::unorm(v, if qfmt == 0 { 8 } else { 16 }) as f32 / if qfmt == 0 { 255.0 } else { 65535.0 }); }
                        for c in 0..4 { acc[c] += p[c] * wt; }
                    }
                    row[bx] = acc;
                }
            });
            b
        } else if dmb == "pair" || dmb == "all" {
            box_pass_parity(&sheared, &blur_taps(radius, 0), &blur_taps(radius, 1), dir, bwrap)
        } else {
            box_pass(&sheared, radius, dir, bwrap)
        };
        o.px.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let s = if y_major {
                    (y as f32 + 0.5 - h as f32 * 0.5) * dx / dy
                } else {
                    (x as f32 + 0.5 - w as f32 * 0.5) * dy / dx
                };
                let (u, v) = if ep == 0 {
                    if y_major {
                        ((x as f32 + 0.5 - s) / w as f32, (y as f32 + 0.5) / h as f32)
                    } else {
                        ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5 - s) / h as f32)
                    }
                } else if y_major {
                    ((x as f32 + ep as f32 + 0.5 - s) / bw as f32, (y as f32 + 0.5) / h as f32)
                } else {
                    ((x as f32 + 0.5) / w as f32, (y as f32 + ep as f32 + 0.5 - s) / bh as f32)
                };
                row[x] = tex(&blurred, u, v);
            }
        });
        o
    }

    fn gradient_map(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let src = self.input_at(n.inputs[0], w, h);
        let v8w_param = self.param(n, "v8word", seed).and_then(|v| v.first().copied()).unwrap_or(0.0);
        let p = self.p;
        let b3 = n.b3;
        let floats_keys = if p.version >= 8 { b3 & 4 != 0 } else { b3 & 2 != 0 };
        let has_v8w = p.version >= 8 && b3 & 2 != 0;
        let repeat = crate::envc::var("SDRT_GMREP").as_deref() != Ok("0")
            && if p.version >= 8 { b3 & 8 != 0 } else { b3 & 4 != 0 };
        let mid = b3 & 1 != 0;
        let keys = self.gradient_keys(n);
        if crate::envc::var("SDRT_GMDBG").is_ok() { eprintln!("gm node {} keys (float {} mid {} rep {} v{} b3 {:#x} v8word {:?}): {:?}", n.index, floats_keys, mid, repeat, p.version, n.b3, n.param("v8word"), keys); }
        // keys interpolate linearly when byte3 bit 0 (midpoints) is set, with smoothstep otherwise
        // (the midpoint value has no effect); v8+ files interpolate linearly without the bit
        // v8+ with byte3 bit 0: smoothstep too (nian-lun 213, keys 0/0.33/0.67/1: input 0.9626 →
        // 0.9647 = smoothstep, linear gave 0.886; 213/243 0.098 → 1/255). SDRT_GMV8MID=linear: old
        let smooth = if p.version >= 8 && mid { crate::env_once!("SDRT_GMV8MID").as_deref() != Ok("linear") } else { !mid && p.version < 8 };
        let v8w = if has_v8w && crate::envc::var("SDRT_GMV8").as_deref() != Ok("0") { v8w_param } else { 0.0 };
        let mode = crate::envc::var("SDRT_GMLUT");
        // Default LUT storage (cx/gmv8_2/REPORT.md, diag_gm_edge):
        // - v8+ with v8word (byte3 bit 1), any node format: unclamped 1024-texel f32 LUT built by
        //   `build_v8_lut` (phantom flat endpoints, EPS = 0.5/1023 key snapping, de Casteljau +
        //   warm-started Newton/midpoint solver), quantized once at node output.
        // - 16F/32F without v8word (fmt >= 2): unclamped 1024-texel f32 LUT (16F outputs truncate
        //   toward zero on the render target).
        // - UNORM8/16 without v8word (fmt <= 1): 256-texel UNORM16 LUT interpolated in 16-bit ints.
        // SDRT_GMLUT=0 evaluates keys directly; SDRT_GMLUT=<N> forces an N-texel UNORM16 LUT.
        if mode.is_err() && (has_v8w || n.format() >= 2) {
            let lut_f32 = gradient_lut_f32(&keys, has_v8w, v8w, smooth);
            let is_f16 = n.format() == 2;
            let mut out = Image::new(w, h, n.color, [0.0; 4]);
            for (dst, s) in out.px.iter_mut().zip(&src.px) {
                let x = if src.color { s[3] } else { s[0] };
                let mut c = sample_f32_lut(&lut_f32, x, repeat);
                if is_f16 {
                    c = c.map(crate::image::half_trunc);
                }
                *dst = if n.color { c } else { Image::gray(c[0]) };
            }
            return out;
        }
        let lut: Option<Vec<[f32; 4]>> = (mode.as_deref() == Ok("old"))
            .then(|| (0..256).map(|i| lookup(&keys, i as f32 / 255.0, smooth)).collect());
        let n_lut: Option<usize> = match mode.as_deref() {
            Ok("0") | Ok("old") => None,
            Ok(s) => s.parse::<usize>().ok().filter(|&nl| nl >= 2),
            Err(_) => Some(256),
        };
        let lut16: Option<Vec<[u32; 4]>> = n_lut.map(|nl| gradient_lut16(&keys, v8w, smooth, nl));
        let mut out = Image::new(w, h, n.color, [0.0; 4]);
        for k in 0..w * h {
            let s = src.px[k];
            // measured: a color input is mapped by its alpha
            let x_raw = if src.color { s[3] } else { s[0] };
            let x = if x_raw.is_finite() { x_raw } else { 0.0 };
            if let Some(t) = &lut16 {
                let nl = t.len();
                let nf = nl as f32;
                let (i0, i1, f) = if repeat {
                    let coord = x.mul_add((nf - 1.0) / nf, 0.5 / nf).rem_euclid(1.0);
                    let u = coord * nf - 0.5;
                    if u < 0.0 {
                        (nl - 1, 0, u + 1.0)
                    } else if u >= (nl - 1) as f32 {
                        (nl - 1, 0, u - (nl - 1) as f32)
                    } else {
                        let i = u.floor() as usize;
                        (i, i + 1, u - i as f32)
                    }
                } else {
                    let coord = x.clamp(0.0, 1.0).mul_add((nf - 1.0) / nf, 0.5 / nf);
                    let u = coord * nf - 0.5;
                    let i = (u.floor().max(0.0) as usize).min(nl - 2);
                    (i, i + 1, u - i as f32)
                };
                let wt = ((f * 256.0) as f64 + 0.5).floor().clamp(0.0, 256.0) as u32;
                let c: [f32; 4] = std::array::from_fn(|ch| ((t[i0][ch] * (256 - wt) + t[i1][ch] * wt + 128) >> 8) as f32 / 65535.0);
                if crate::envc::var("SDRT_GMDBG").as_deref() == Ok("px") && [29usize, 376, 419, 595, 648, 709, 776, 985].contains(&k) { eprintln!("gm px {} x {:e} ({:.6}) i0 {} i1 {} wt {} T {} {} -> {}", k, x, x * (nf - 1.0), i0, i1, wt, t[i0][0], t[i1][0], c[0] * 65535.0); }
                out.px[k] = if n.color { c } else { Image::gray(c[0]) };
                continue;
            }
            out.px[k] = match &lut {
                Some(t) => {
                    let u = x.clamp(0.0, 1.0) * 255.0;
                    let i = (u.floor() as usize).min(254);
                    let f = ((u - i as f32) * 256.0).round() / 256.0;
                    std::array::from_fn(|c| lerp(t[i][c], t[i + 1][c], f))
                }
                None => {
                    let xq = if repeat { x.rem_euclid(1.0) } else { x.clamp(0.0, 1.0) };
                    let c = if v8w != 0.0 {
                        lookup_smooth(&keys, xq as f64, v8w as f64).map(|v| v as f32)
                    } else {
                        lookup(&keys, xq, smooth)
                    };
                    if n.color { c } else { Image::gray(c[0]) }
                }
            };
        }
        out
    }

    /// A gradient map's keys, sorted by position: float keys (byte3 bit 2 from v8 on, bit 1
    /// before) or 16-bit positions with 8-bit colours / 16-bit grays; byte3 bit 0 adds a
    /// midpoint to each key (shared by the CPU path and try_gpu_gradient_map).
    fn gradient_keys(&self, n: &Node) -> Vec<(f32, [f32; 4])> {
        let p = self.p;
        let b3 = n.b3;
        let floats_keys = if p.version >= 8 { b3 & 4 != 0 } else { b3 & 2 != 0 };
        let mid = b3 & 1 != 0;
        let mut keys: Vec<(f32, [f32; 4])> = vec![];
        let mut o = n.data;
        for _ in 0..n.count {
            if floats_keys {
                let pos = p.f(o);
                let c = if n.color { [p.f(o + 4), p.f(o + 8), p.f(o + 12), p.f(o + 16)] } else { Image::gray(p.f(o + 4)) };
                keys.push((pos, c));
                o += 4 * ((if n.color { 5 } else { 2 }) + mid as usize);
            } else {
                let b = p.bytes(o, 8);
                let pos = u16::from_le_bytes([b[0], b[1]]) as f32 / 65535.0;
                let c = if n.color {
                    [b[2] as f32 / 255.0, b[3] as f32 / 255.0, b[4] as f32 / 255.0, b[5] as f32 / 255.0]
                } else {
                    Image::gray(u16::from_le_bytes([b[2], b[3]]) as f32 / 65535.0)
                };
                keys.push((pos, c));
                o += (if n.color { 6 } else { 4 }) + if mid { 2 } else { 0 };
            }
        }

        keys.sort_by(|a, b| a.0.total_cmp(&b.0));
        keys
    }

    /// Supply the image of image input `uid` (any size, RGBA f32 as decoded from the file).
    /// `format` is the storage of the file: 0 8-bit, 1 16-bit, 3 float.
    pub fn set_image(&mut self, uid: u32, img: Image, format: u32) {
        self.image_inputs.insert(uid, (std::sync::Arc::new(img), format));
    }

    fn bitmap(&mut self, n: &Node) -> Image {
        let (w, h) = n.size();
        let p = self.p;
        if n.b3 == 0 {
            if let Some((img, sf)) = self.image_inputs.get(&(n.data as u32)).cloned() {
                return input_image(&img, sf, n.color, n.format(), w, h);
            }
            // graph input image: no image is given, use its default (v8+) or transparent black
            let v = self.inputs.get(&(n.data as u32)).map(|v| v.fv()).unwrap_or([0.0; 4]);
            return Image::new(w, h, n.color, if n.color { v } else { Image::gray(v[0]) });
        }
        if n.data == 0 {
            // missing embedded bitmap resource (null block offset, e.g. test-2-twm node 800): black
            return Image::new(w, h, n.color, if n.color { [0.0; 4] } else { Image::gray(0.0) });
        }
        let b3 = n.b3;
        if b3 & 8 != 0 {
            let len = p.w(n.data) as usize;
            let jpg = p.bytes(n.data + 4, len);
            let mut img = match decode_jpeg(jpg) {
                Ok(d) => from_dynamic(&d, n.color),
                Err(e) => {
                    self.unsupported(n.index, format!("jpeg: {e}"));
                    return Image::new(w, h, n.color, [0.0; 4]);
                }
            };
            if n.data2 != 0 {
                let len2 = p.w(n.data2) as usize;
                if let Ok(a) = decode_jpeg(p.bytes(n.data2 + 4, len2)) {
                    let a = from_dynamic(&a, false);
                    for (q, s) in img.px.iter_mut().zip(&a.px) {
                        q[3] = s[0];
                    }
                }
            }
            return img.resized(w, h);
        }
        let ch = match b3 & 3 {
            1 => 1,
            2 => 3,
            _ => 4,
        };
        let bpc = if b3 & 0x20 != 0 { 4 } else if b3 & 0x14 != 0 { 2 } else { 1 };
        let raw = p.bytes(n.data, w * h * ch * bpc);
        let mut img = Image::new(w, h, n.color, [0.0; 4]);
        for k in 0..w * h {
            let mut c = [0.0, 0.0, 0.0, 1.0];
            for j in 0..ch {
                let o = (k * ch + j) * bpc;
                c[j] = match bpc {
                    4 => f32::from_le_bytes(raw[o..o + 4].try_into().unwrap()),
                    2 if b3 & 0x10 != 0 => f16_to_f32(u16::from_le_bytes([raw[o], raw[o + 1]])),
                    2 => u16::from_le_bytes([raw[o], raw[o + 1]]) as f32 / 65535.0,
                    _ => raw[o] as f32 / 255.0,
                };
            }
            if ch == 1 {
                c = Image::gray(c[0]);
            }
            img.px[k] = c;
        }
        img
    }

    fn pixel_processor(&mut self, n: &Node, seed: u32) -> Image {
        let (w, h) = n.size();
        let f = self.func(n.data);
        if n.flags >> 16 != 0 {
            // value processor: one evaluation, read by others with op 3
            let mut vars = std::mem::take(&mut self.node_vars);
            let srcs: Vec<std::sync::Arc<Image>> = n.inputs.iter().filter_map(|&k| if k < self.images.len() { self.images[k].clone().or_else(|| Some(std::sync::Arc::new(Image::new(1, 1, true, [0.0; 4])))) } else { Some(std::sync::Arc::new(Image::new(1, 1, true, [0.0; 4]))) }).collect();
            let fmts: Vec<u32> = n.inputs.iter().map(|&k| if k < self.p.nodes.len() { self.tex_format(k) } else { 3 }).collect();
            if crate::envc::var("SDRT_VPLOG").is_ok() { eprintln!("vp {} inputs {:?} present {:?}", n.index, n.inputs, n.inputs.iter().map(|&k| self.images.get(k).is_some_and(|i| i.is_some())).collect::<Vec<_>>()); }
            let old = crate::env_once!("SDRT_VPRAND").as_deref() == Ok("node");
            let state = if old { seed & M24 } else { pixel_seed(seed, [0.0, 0.0]) };
            let mut env = NodeEnv { rt: self, state, pos: [0.5, 0.5], number: 0.0, size: [w as f32, h as f32], writes: Vec::new(), srcs, fmts, vp: !old, filter: 0 };
            let v = Eval::run_with_vars(&f, &mut env, &mut vars);
            let writes = std::mem::take(&mut env.writes);
            self.node_vars = vars;
            if crate::envc::var("SDRT_VPLOG").is_ok() {
                eprintln!("vplog node {} value {:?} writes {:?}", n.index, v, writes);
            }
            for (slot, w) in writes {
                self.vp_slots.insert(slot, w);
            }
            self.vp.insert(n.index, v);
            return Image::new(1, 1, n.color, [0.0; 4]);
        }
        let srcs: Vec<std::sync::Arc<Image>> = n.inputs.iter().map(|&k| if k < self.images.len() { self.images[k].clone().unwrap_or_else(|| std::sync::Arc::new(Image::new(1, 1, true, [0.0; 4]))) } else { std::sync::Arc::new(Image::new(1, 1, true, [0.0; 4])) }).collect();
        let node_vars = self.node_vars.clone();
        let refs: Vec<&Image> = srcs.iter().map(|a| a.as_ref()).collect();
        let fmts: Vec<u32> = n.inputs.iter().map(|&k| if k < self.p.nodes.len() { self.tex_format(k) } else { 3 }).collect();
        // samples tile on both axes whatever byte3 says (scratch/pp2/samp3.py)
        let wrap = (true, true);
        let rt: &Runtime = self;
        // registers that are the same at every pixel: computed at the first pixel, then reused
        let inv = f.pixel_invariant();
        let prefill = {
            let pos = [0.5 / w as f32, 0.5 / h as f32];
            let mut env = PixEnv { rt, size: [w as f32, h as f32], srcs: &refs, fmts: &fmts, wrap, pos, state: pixel_seed(seed, pos), written: None, filter: 0 };
            Eval::run_capture(&f, &mut env, &mut node_vars.clone(), &inv).1
        };
        use rayon::prelude::*;
        let rows: Vec<Vec<[f32; 4]>> = (0..h)
            .into_par_iter()
            .map(|y| {
                (0..w)
                    .map(|x| {
                        let pos = [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32];
                        let mut env = PixEnv { rt, size: [w as f32, h as f32], srcs: &refs, fmts: &fmts, wrap, pos, state: pixel_seed(seed, pos), written: None, filter: 0 };
                        let v = Eval::run_prefilled(&f, &mut env, &mut node_vars.clone(), &prefill);
                        let c = v.fv();
                        if n.color {
                            if v.n() >= 4 { c } else if v.n() == 3 { [c[0], c[1], c[2], 1.0] } else { [c[0], c[0], c[0], 1.0] }
                        } else {
                            Image::gray(c[0])
                        }
                    })
                    .collect()
            })
            .collect();
        Image { w, h, color: n.color, px: rows.into_iter().flatten().collect() }
    }
}

/// 1-D squared distance transform (Felzenszwalb–Huttenlocher) of f over n samples; also returns,
/// for each sample, the index of the sample its minimum comes from.
fn dt1(f: &[f32], out: &mut [f32], arg: &mut [usize]) {
    let n = f.len();
    let mut v = vec![0usize; n];
    let mut z = vec![0f32; n + 1];
    let mut k = 0usize;
    let mut first = None;
    for q in 0..n {
        if f[q] >= 1e19 {
            continue;
        }
        match first {
            None => {
                first = Some(q);
                v[0] = q;
                z[0] = f32::NEG_INFINITY;
                z[1] = f32::INFINITY;
            }
            Some(_) => loop {
                let p = v[k];
                let sx = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2.0 * (q as f32 - p as f32));
                if sx <= z[k] {
                    if k == 0 {
                        v[0] = q;
                        z[0] = f32::NEG_INFINITY;
                        z[1] = f32::INFINITY;
                        break;
                    }
                    k -= 1;
                } else {
                    k += 1;
                    v[k] = q;
                    z[k] = sx;
                    z[k + 1] = f32::INFINITY;
                    break;
                }
            },
        }
    }
    if first.is_none() {
        out.iter_mut().for_each(|o| *o = 1e20);
        arg.iter_mut().for_each(|a| *a = usize::MAX);
        return;
    }
    let mut j = 0;
    for q in 0..n {
        while z[j + 1] < q as f32 {
            j += 1;
        }
        let p = v[j];
        out[q] = (q as f32 - p as f32).powi(2) + f[p];
        arg[q] = p;
    }
}

/// Squared Euclidean distance to the nearest true pixel and that pixel's index; wrapping works on
/// lines tripled end to end.
/// An embedded JPEG as sbsrender decodes it (crate::jpeg); files it can't take (progressive…)
/// fall back to the image crate.
fn decode_jpeg(b: &[u8]) -> Result<image::DynamicImage, String> {
    match crate::jpeg::decode(b) {
        Ok(d) if d.ch == 1 => Ok(image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(d.w as u32, d.h as u32, d.px).unwrap())),
        Ok(d) => Ok(image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(d.w as u32, d.h as u32, d.px).unwrap())),
        Err(_) => image::load_from_memory_with_format(b, image::ImageFormat::Jpeg).map_err(|e| e.to_string()),
    }
}

/// Squared distances with per-axis weights: d² = (sx·dx)² + (sy·dy)².
fn edt(inside: &[bool], w: usize, h: usize, wrap: bool, sx: f32, sy: f32) -> (Vec<f32>, Vec<usize>) {
    const INF: f32 = 1e20;
    let rep = if wrap { 3 } else { 1 };
    let off = |n: usize| if wrap { n } else { 0 };
    // columns
    let mut col_d = vec![INF; w * h];
    let mut col_r = vec![usize::MAX; w * h];
    let (mut f, mut o, mut a) = (vec![0f32; h * rep], vec![0f32; h * rep], vec![0usize; h * rep]);
    for x in 0..w {
        for i in 0..h * rep {
            f[i] = if inside[(i % h) * w + x] { 0.0 } else { INF };
        }
        dt1(&f, &mut o, &mut a);
        for y in 0..h {
            let i = y + off(h);
            col_d[y * w + x] = o[i];
            col_r[y * w + x] = if a[i] == usize::MAX { usize::MAX } else { a[i] % h };
        }
    }
    // rows
    let mut d = vec![INF; w * h];
    let mut near = vec![usize::MAX; w * h];
    let (mut f, mut o, mut a) = (vec![0f32; w * rep], vec![0f32; w * rep], vec![0usize; w * rep]);
    for y in 0..h {
        // weighted: min_p f[p]·sy² + sx²(q − p)² = sx²·min_p (f[p]·sy²/sx² + (q − p)²)
        let ratio = (sy * sy) / (sx * sx);
        for i in 0..w * rep {
            let c = col_d[y * w + i % w];
            f[i] = if c >= INF { INF } else { c * ratio };
        }
        dt1(&f, &mut o, &mut a);
        for x in 0..w {
            let i = x + off(w);
            d[y * w + x] = if o[i] >= 1e19 { INF } else { o[i] * sx * sx };
            if a[i] != usize::MAX {
                let c = a[i] % w;
                let r = col_r[y * w + c];
                near[y * w + x] = if r == usize::MAX { usize::MAX } else { r * w + c };
            }
        }
    }
    (d, near)
}

/// y of a curve at x: the segment holding x, its cubic Bézier solved for t by bisection. Before the
/// first point and after the last, SD adds end points at x = 0 and 1 with the first/last y, the
/// handles as controls (measured, lab/py/curveends.py).
fn bezier_curve(pts: &[[f32; 6]], x: f32) -> f32 {
    if pts.is_empty() {
        return x;
    }
    let seg = |p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]| -> f32 {
        let bz = |t: f32, i: usize| {
            let u = 1.0 - t;
            u * u * u * p0[i] + 3.0 * u * u * t * p1[i] + 3.0 * u * t * t * p2[i] + t * t * t * p3[i]
        };
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if bz(mid, 0) < x { lo = mid } else { hi = mid }
        }
        bz(0.5 * (lo + hi), 1)
    };
    let first = pts[0];
    if x <= first[0] {
        let s = [0.0, first[1]];
        return seg(s, s, [first[2], first[3]], [first[0], first[1]]);
    }
    for s in pts.windows(2) {
        let (a, b) = (s[0], s[1]);
        if x <= b[0] {
            // P0 = a.pos, P1 = a.right handle, P2 = b.left handle, P3 = b.pos
            return seg([a[0], a[1]], [a[4], a[5]], [b[2], b[3]], [b[0], b[1]]);
        }
    }
    let last = pts[pts.len() - 1];
    let e = [1.0, last[1]];
    seg([last[0], last[1]], [last[4], last[5]], e, e)
}

/// Taps of a box of half-width r: each integer offset weighs its pixel's overlap with [−r, r].
fn box_taps(r: f32) -> Vec<(i64, f32)> {
    if r <= 0.5 {
        return vec![(0, 1.0)];
    }
    let k = (r + 0.5).ceil() as i64;
    let mut taps = vec![];
    for t in -k..=k {
        let lo = (t as f32 - 0.5).max(-r);
        let hi = (t as f32 + 0.5).min(r);
        if hi > lo {
            taps.push((t, (hi - lo) / (2.0 * r)));
        }
    }
    taps
}

/// The blur's box taps as the reference GPU applies them, for an output pixel of the given parity
/// along the pass: texels are fetched in bilinear pairs whose fraction is rounded to 1/256
/// (scratch/fb/blurkern*.py, blurfit.py; impulse responses exact for r = 1.75 … 12 at even and odd
/// positions). Five taps (r < 2.5): pairs symmetric about the centre, (1, 2) and (−1, −2). Seven or
/// more: texture-aligned pairs (2m, 2m+1) — only an end pair holding a partial outer tap and a full
/// inner one changes, the left end for odd pixels and the right end for even ones (k odd).
/// SDRT_BLURBOX=1: plain box (test switch).
fn blur_taps(r: f32, parity: i64) -> Vec<(i64, f32)> {
    let mut taps = box_taps(r);
    let k = taps.iter().map(|&(t, _)| t.abs()).max().unwrap_or(0);
    if crate::envc::var("SDRT_BLURBOX").is_ok() || k < 2 {
        return taps;
    }
    let get = |taps: &Vec<(i64, f32)>, t: i64| taps.iter().find(|&&(x, _)| x == t).map(|&(_, v)| v);
    let pair = |taps: &mut Vec<(i64, f32)>, first: i64, second: i64| {
        if let (Some(a), Some(b)) = (get(taps, first), get(taps, second)) {
            let s = a + b;
            let q = (b / s * 256.0).round() / 256.0;
            for e in taps.iter_mut() {
                if e.0 == first {
                    e.1 = s * (1.0 - q);
                } else if e.0 == second {
                    e.1 = s * q;
                }
            }
        }
    };
    if k == 2 {
        pair(&mut taps, 1, 2);
        pair(&mut taps, -1, -2);
    } else {
        // texel j + t is the first of its pair when j + t is even
        for t in -k..k {
            if (parity + t).rem_euclid(2) == 0 {
                pair(&mut taps, t, t + 1);
            }
        }
    }
    taps
}

/// The blur's fetches for an output pixel of the given parity: (offset in pixels, weight) — single
/// taps at integer offsets, and each pair of `blur_taps` as one bilinear fetch at its fractional
/// position (the same pairing and 1/256 fractions).
fn blur_fetches(r: f32, parity: i64) -> Vec<(f32, f32)> {
    let taps = box_taps(r);
    let k = taps.iter().map(|&(t, _)| t.abs()).max().unwrap_or(0);
    if crate::envc::var("SDRT_BLURBOX").is_ok() || k < 2 {
        return taps.iter().map(|&(t, w)| (t as f32, w)).collect();
    }
    let get = |t: i64| taps.iter().find(|&&(x, _)| x == t).map(|&(_, v)| v);
    let pairs: Vec<(i64, i64)> = if k == 2 { vec![(1, 2), (-1, -2)] } else { (-k..k).filter(|t| (parity + t).rem_euclid(2) == 0).map(|t| (t, t + 1)).collect() };
    let mut used = vec![];
    let mut out = vec![];
    for (a, b) in pairs {
        if let (Some(wa), Some(wb)) = (get(a), get(b)) {
            let s = wa + wb;
            let q = (wb / s * 256.0).round() / 256.0;
            out.push((a as f32 + q * (b - a) as f32, s));
            used.push(a);
            used.push(b);
        }
    }
    for &(t, w) in &taps {
        if !used.contains(&t) {
            out.push((t as f32, w));
        }
    }
    out
}

/// A separable pass whose taps depend on the parity of the output coordinate along `dir`.
fn box_pass_parity(src: &Image, even: &[(i64, f32)], odd: &[(i64, f32)], dir: (i64, i64), wrap: (bool, bool)) -> Image {
    let mut o = Image::new(src.w, src.h, src.color, [0.0; 4]);
    for y in 0..src.h as i64 {
        for x in 0..src.w as i64 {
            let taps = if (x * dir.0 + y * dir.1) % 2 == 0 { even } else { odd };
            let mut acc = [0.0f32; 4];
            for &(t, wt) in taps {
                let p = src.fetch(x + t * dir.0, y + t * dir.1, wrap);
                for c in 0..4 {
                    acc[c] += p[c] * wt;
                }
            }
            o.px[y as usize * src.w + x as usize] = acc;
        }
    }
    o
}

fn box_pass(src: &Image, r: f32, dir: (i64, i64), wrap: (bool, bool)) -> Image {
    box_pass_taps(src, &box_taps(r), dir, wrap)
}

fn box_pass_taps(src: &Image, taps: &[(i64, f32)], dir: (i64, i64), wrap: (bool, bool)) -> Image {
    let taps = taps.to_vec();
    let mut o = Image::new(src.w, src.h, src.color, [0.0; 4]);
    for y in 0..src.h as i64 {
        for x in 0..src.w as i64 {
            let mut acc = [0.0f32; 4];
            for &(t, wt) in &taps {
                let p = src.fetch(x + t * dir.0, y + t * dir.1, wrap);
                for c in 0..4 {
                    acc[c] += p[c] * wt;
                }
            }
            o.px[y as usize * src.w + x as usize] = acc;
        }
    }
    o
}

/// A gradient map's 1024-texel unclamped f32 table (v8word, or a float node format): the v8+
/// Bézier table, or the gradient at i/1023 in f64 (shared by the CPU and GPU paths).
fn gradient_lut_f32(keys: &[(f32, [f32; 4])], has_v8w: bool, v8w: f32, smooth: bool) -> [[f32; 4]; 1024] {
    if has_v8w {
        build_v8_lut(keys, v8w)
    } else {
        std::array::from_fn(|i| lookup64(keys, i as f64 / 1023.0, smooth).map(|v| v as f32))
    }
}

/// A gradient map's nl-texel UNORM16 table (UNORM node formats without v8word): the gradient
/// at i/(nl - 1), rounded to 16-bit codes (shared by the CPU and GPU paths).
fn gradient_lut16(keys: &[(f32, [f32; 4])], v8w: f32, smooth: bool, nl: usize) -> Vec<[u32; 4]> {
    let denom = (nl - 1) as f64;
    (0..nl).map(|i| (if v8w != 0.0 { lookup_smooth(keys, i as f64 / denom, v8w as f64) } else { lookup64(keys, i as f64 / denom, smooth) })
        .map(|v| (((v as f32) * 65535.0).clamp(0.0, 65535.0) as f64 + 0.5).floor() as u32)).collect()
}

/// Build the exact 1024-texel unclamped f32 LUT for v8+ gradient maps with `v8word` (byte3 bit 1),
/// verified to ≤ 1 f32 ULP across 55 full/zoom probes and edge cases on sbsrender 14.1 vk
/// (cx/gmv8_2/REPORT.md). Keys must already be sorted by position.
fn build_v8_lut(sorted_keys: &[(f32, [f32; 4])], w: f32) -> [[f32; 4]; 1024] {
    const STEP: f32 = 1.0 / 1023.0;
    const EPS: f32 = 0.5 * STEP;

    #[inline]
    fn mix(a: f32, b: f32, t: f32) -> f32 {
        a * (1.0 - t) + b * t
    }
    #[inline]
    fn cubic(p: [f32; 4], t: f32) -> f32 {
        let a = mix(p[0], p[1], t);
        let b = mix(p[1], p[2], t);
        let c = mix(p[2], p[3], t);
        mix(mix(a, b, t), mix(b, c, t), t)
    }
    #[inline]
    fn derivative(p: [f32; 4], t: f32) -> f32 {
        let a = 3.0 * (p[1] - p[0]);
        let b = 3.0 * (p[2] - p[1]);
        let c = 3.0 * (p[3] - p[2]);
        mix(mix(a, b, t), mix(b, c, t), t)
    }
    fn invert(p: [f32; 4], x: f32, mut t: f32) -> f32 {
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for outer in 0..32 {
            let guess = t;
            for _ in 0..100 {
                let err = cubic(p, t) - x;
                if err.abs() < EPS {
                    return t;
                }
                let d = derivative(p, t);
                if d.abs() < 1.0e-4 {
                    break;
                }
                t -= err / d;
                if !t.is_finite() {
                    break;
                }
            }
            t = guess;
            if outer != 0 && cubic(p, hi) - cubic(p, lo) < EPS {
                return t;
            }
            if cubic(p, t) < x {
                lo = t;
            } else {
                hi = t;
            }
            t = (lo + hi) * 0.5;
        }
        t
    }

    if sorted_keys.is_empty() {
        return [[0.0, 0.0, 0.0, 1.0]; 1024];
    }
    if sorted_keys.len() == 1 {
        return [sorted_keys[0].1; 1024];
    }
    let first = sorted_keys[0];
    let last = sorted_keys[sorted_keys.len() - 1];
    let mut keys = Vec::with_capacity(sorted_keys.len() + 2);
    if first.0 > 0.0 {
        keys.push((0.0, first.1));
    }
    keys.extend_from_slice(sorted_keys);
    if last.0 < 1.0 {
        keys.push((1.0, last.1));
    }
    let n = keys.len();
    let mut left = keys.clone();
    let mut right = keys.clone();
    if n > 2 {
        for j in 1..n - 1 {
            let span = keys[j + 1].0 - keys[j - 1].0;
            let dl = w * (keys[j].0 - keys[j - 1].0) * 0.5;
            let dr = w * (keys[j + 1].0 - keys[j].0) * 0.5;
            left[j].0 -= dl;
            right[j].0 += dr;
            for c in 0..4 {
                let slope = if span != 0.0 { (keys[j + 1].1[c] - keys[j - 1].1[c]) / span } else { 0.0 };
                left[j].1[c] -= dl * slope;
                right[j].1[c] += dr * slope;
            }
        }
        right[0].0 = keys[0].0 + (left[1].0 - keys[0].0) * w * 0.5;
        left[n - 1].0 = keys[n - 1].0 + (right[n - 2].0 - keys[n - 1].0) * w * 0.5;
        for c in 0..4 {
            right[0].1[c] = keys[0].1[c] + (left[1].1[c] - keys[0].1[c]) * w * 0.5;
            left[n - 1].1[c] = keys[n - 1].1[c] + (right[n - 2].1[c] - keys[n - 1].1[c]) * w * 0.5;
        }
    }
    let (mut j, mut t) = (0usize, 0.5f32);
    std::array::from_fn(|i| {
        let x = i as f32 * STEP;
        if j < n - 2 && x > keys[j + 1].0 + EPS {
            j += 1;
            t = 0.5;
        }
        let (a, b) = (keys[j], keys[j + 1]);
        if (x - a.0).abs() <= EPS {
            return a.1;
        }
        if (x - b.0).abs() <= EPS {
            return b.1;
        }
        if x <= first.0 {
            return first.1;
        }
        if x >= last.0 {
            return last.1;
        }
        if w == 0.0 || n == 2 || a.0 == b.0 {
            return std::array::from_fn(|c| a.1[c] + (x - a.0) * ((b.1[c] - a.1[c]) / (b.0 - a.0)));
        }
        let px = [a.0, right[j].0, left[j + 1].0, b.0];
        t = invert(px, x, t);
        std::array::from_fn(|c| cubic([a.1[c], right[j].1[c], left[j + 1].1[c], b.1[c]], t))
    })
}

/// Sample a 1024-texel f32 LUT with GPU texture-unit 1/256 fractional weights.
fn sample_f32_lut(lut: &[[f32; 4]; 1024], x: f32, repeat: bool) -> [f32; 4] {
    let x = if x.is_finite() { x } else { 0.0 };
    let coord = if repeat {
        x.mul_add(1023.0 / 1024.0, 0.5 / 1024.0).rem_euclid(1.0)
    } else {
        x.clamp(0.0, 1.0).mul_add(1023.0 / 1024.0, 0.5 / 1024.0)
    };
    let u = coord * 1024.0 - 0.5;
    let i = u.floor() as i32;
    let wt = ((u - i as f32) * 256.0 + 0.5).floor() / 256.0;
    let index = |v: i32| if repeat { v.rem_euclid(1024) as usize } else { v.clamp(0, 1023) as usize };
    let (p0, p1) = (lut[index(i)], lut[index(i + 1)]);
    if wt <= 0.0 {
        return p0;
    }
    if wt >= 1.0 {
        return p1;
    }
    let wf = wt as f64;
    std::array::from_fn(|c| (p0[c] as f64 * (1.0 - wf) + p1[c] as f64 * wf) as f32)
}

/// Whether the GPU's integer reproduction of sample_f32_lut's f64 mix (gpu_color_common.wgsl
/// "mix_f64") covers this table: finite texels that are zero or at least 2^-95 in magnitude
/// (no subnormal result is possible then), and neighbouring finite non-zero texels (every pair
/// a lookup can mix, with the wrap-around pair when repeating) whose exponents differ by at
/// most 31 (their aligned sum fits 64 bits). Infinities and NaNs (build_v8_lut between keys at
/// one position) are reproduced, except two NaNs of different payloads side by side (which one
/// the f64 addition propagates depends on the compiler's operand order).
#[cfg(feature = "gpu")]
fn gpu_mixable(lut: &[[f32; 4]], repeat: bool) -> Result<(), String> {
    let tiny = f32::from_bits(32 << 23);
    let exponent = |v: f32| ((v.to_bits() >> 23) & 255) as i32;
    if let Some((i, v)) = lut.iter().flatten().enumerate().find(|&(_, &v)| v.is_finite() && v != 0.0 && v.abs() < tiny) {
        return Err(format!("texel {} channel {}: {v:e}", i / 4, i % 4));
    }
    let n = lut.len();
    let pairs = (0..n - 1).map(|i| (i, i + 1)).chain(repeat.then_some((n - 1, 0)));
    for (i, j) in pairs {
        for c in 0..4 {
            let (a, b) = (lut[i][c], lut[j][c]);
            if a.is_nan() && b.is_nan() && (a.to_bits() | 0x40_0000) != (b.to_bits() | 0x40_0000) {
                return Err(format!("texels {i}/{j} channel {c}: two different NaNs"));
            }
            if a.is_finite() && b.is_finite() && a != 0.0 && b != 0.0 && (exponent(a) - exponent(b)).abs() > 31 {
                return Err(format!("texels {i}/{j} channel {c}: {a:e} next to {b:e}"));
            }
        }
    }
    Ok(())
}

/// v8+ "smooth" gradient (direct f64 fallback for SDRT_GMLUT=0).
fn lookup_smooth(keys: &[(f32, [f32; 4])], x: f64, w: f64) -> [f64; 4] {
    let n = keys.len();
    if n == 0 { return [0.0, 0.0, 0.0, 1.0]; }
    let kx: Vec<f64> = keys.iter().map(|k| k.0 as f64).collect();
    let ky: Vec<[f64; 4]> = keys.iter().map(|k| k.1.map(|v| v as f64)).collect();
    if n == 1 || x <= kx[0] { return ky[0]; }
    if x >= kx[n - 1] { return ky[n - 1]; }
    // handles: (x, y[4]) left and right of each key
    let mut hl: Vec<(f64, [f64; 4])> = (0..n).map(|i| (kx[i], ky[i])).collect();
    let mut hr = hl.clone();
    for i in 1..n - 1 {
        let span = kx[i + 1] - kx[i - 1];
        let slope: [f64; 4] = std::array::from_fn(|c| if span > 0.0 { (ky[i + 1][c] - ky[i - 1][c]) / span } else { 0.0 });
        let dl = w * (kx[i] - kx[i - 1]) / 2.0;
        let dr = w * (kx[i + 1] - kx[i]) / 2.0;
        hl[i] = (kx[i] - dl, std::array::from_fn(|c| ky[i][c] - dl * slope[c]));
        hr[i] = (kx[i] + dr, std::array::from_fn(|c| ky[i][c] + dr * slope[c]));
    }
    if n == 2 {
        for c in [0usize, 1] { let _ = c; }
        hr[0] = (kx[0] + (kx[1] - kx[0]) / 3.0, std::array::from_fn(|c| ky[0][c] + (ky[1][c] - ky[0][c]) / 3.0));
        hl[1] = (kx[1] - (kx[1] - kx[0]) / 3.0, std::array::from_fn(|c| ky[1][c] - (ky[1][c] - ky[0][c]) / 3.0));
    } else {
        let f = w / 2.0;
        let a = hl[1];
        hr[0] = (kx[0] + (a.0 - kx[0]) * f, std::array::from_fn(|c| ky[0][c] + (a.1[c] - ky[0][c]) * f));
        let b = hr[n - 2];
        hl[n - 1] = (kx[n - 1] + (b.0 - kx[n - 1]) * f, std::array::from_fn(|c| ky[n - 1][c] + (b.1[c] - ky[n - 1][c]) * f));
    }
    let bez = |p0: f64, p1: f64, p2: f64, p3: f64, s: f64| (1.0 - s).powi(3) * p0 + 3.0 * (1.0 - s).powi(2) * s * p1 + 3.0 * (1.0 - s) * s * s * p2 + s.powi(3) * p3;
    for i in 0..n - 1 {
        if x <= kx[i + 1] {
            let (x0, x1, x2, x3) = (kx[i], hr[i].0, hl[i + 1].0, kx[i + 1]);
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            for _ in 0..60 {
                let m = 0.5 * (lo + hi);
                if bez(x0, x1, x2, x3, m) < x { lo = m } else { hi = m }
            }
            let s = 0.5 * (lo + hi);
            return std::array::from_fn(|c| bez(ky[i][c], hr[i].1[c], hl[i + 1].1[c], ky[i + 1][c], s));
        }
    }
    ky[n - 1]
}

/// The gradient in f64 (key positions and colors as stored).
fn lookup64(keys: &[(f32, [f32; 4])], x: f64, smooth: bool) -> [f64; 4] {
    if keys.is_empty() {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let c64 = |c: [f32; 4]| c.map(|v| v as f64);
    if x <= keys[0].0 as f64 {
        return c64(keys[0].1);
    }
    for w in keys.windows(2) {
        let ((p0, c0), (p1, c1)) = ((w[0].0 as f64, c64(w[0].1)), (w[1].0 as f64, c64(w[1].1)));
        if x <= p1 {
            let mut t = if p1 > p0 { (x - p0) / (p1 - p0) } else { 1.0 };
            if smooth {
                t = t * t * (3.0 - 2.0 * t);
            }
            return std::array::from_fn(|k| c0[k] + (c1[k] - c0[k]) * t);
        }
    }
    c64(keys[keys.len() - 1].1)
}

fn lookup(keys: &[(f32, [f32; 4])], x: f32, smooth: bool) -> [f32; 4] {
    if keys.is_empty() {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if x <= keys[0].0 {
        return keys[0].1;
    }
    for w in keys.windows(2) {
        let ((p0, c0), (p1, c1)) = (w[0], w[1]);
        if x <= p1 {
            let mut t = if p1 > p0 { (x - p0) / (p1 - p0) } else { 1.0 };
            if smooth {
                t = t * t * (3.0 - 2.0 * t);
            }
            return [lerp(c0[0], c1[0], t), lerp(c0[1], c1[1], t), lerp(c0[2], c1[2], t), lerp(c0[3], c1[3], t)];
        }
    }
    keys[keys.len() - 1].1
}

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let l = (mx + mn) / 2.0;
    if mx == mn {
        return (0.0, 0.0, l);
    }
    let d = mx - mn;
    // Matches the GPU RGBtoHCV/RGBtoHSL family: both normalisations retain
    // the same epsilon.  The sector offsets below are measured from 32F
    // black-box probes; they account for the particular branchless ordering
    // used by Substance at and around its hue-sector boundaries.
    const E: f32 = 1.0 / 4096.0;
    let s = d / (1.0 - (2.0 * l - 1.0).abs() + E);
    let raw = if mx == r {
        (g - b) + if g < b { 6.0 * d } else { 0.0 }
    } else if mx == g {
        (b - r) + 2.0 * d
    } else {
        (r - g) + 4.0 * d
    };
    let bias = if raw == 0.0 {
        -4.0 * E
    } else if raw < 2.0 * d {
        -2.0 * E
    } else if raw <= 4.0 * d {
        0.0
    } else {
        2.0 * E
    };
    (((raw + bias) / (6.0 * (d + E))).rem_euclid(1.0), s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let f = |mut t: f32| {
        t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    (f(h + 1.0 / 3.0), f(h), f(h - 1.0 / 3.0))
}

/// A caller-supplied image input as the graph-input node (`color`, format `fmt`, w x h) sees it:
/// a gray node reads a color file as the mean of R, G, B (a one-channel file as it is), the alpha
/// is the file's (1 without one); a half-float node stores the values rounded toward zero.
fn input_image(src: &Image, sfmt: u32, color: bool, fmt: u32, w: usize, h: usize) -> Image {
    let third = 1.0f32 / 3.0;
    // a gray node of a colour file resizes the colour image first and takes the mean after
    let late_gray = !color && src.color && crate::envc::var("SDRT_II_GRAY").as_deref() != Ok("early");
    let mut img = Image::new(src.w, src.h, color || late_gray, [0.0; 4]);
    for (q, p) in img.px.iter_mut().zip(&src.px) {
        *q = if color || late_gray {
            *p
        } else if !src.color {
            Image::gray(p[0])
        } else {
            // a dot product with 1/3: G times it, then R and B fused multiply-adds
            Image::gray(p[2].mul_add(third, p[0].mul_add(third, p[1] * third)))
        };
        if fmt == 2 && !late_gray {
            *q = q.map(crate::image::half_trunc);
        }
    }
    let mut out = input_resize(img, sfmt, fmt, w, h);
    if late_gray {
        let mut g = Image::new(out.w, out.h, false, [0.0; 4]);
        for (q, p) in g.px.iter_mut().zip(&out.px) {
            let v = p[2].mul_add(third, p[0].mul_add(third, p[1] * third));
            *q = Image::gray(v);
            if fmt == 2 { *q = q.map(crate::image::half_trunc); }
        }
        out = g;
    }
    out
}

fn input_resize(img: Image, sfmt: u32, fmt: u32, w: usize, h: usize) -> Image {
    if (img.w, img.h) == (w, h) {
        return img;
    }
    let mode = crate::envc::var("SDRT_II_MODE").unwrap_or_default();
    if mode == "resize" {
        return img.resized(w, h);
    }
    // Per axis the file is first resampled to s1 = the node size times a power of two nearest
    // to the file's size (the largest not above it when downsizing, the smallest not below it
    // when upsizing): a bilinear at the new pixel centres from the file held in a power-of-two
    // texture (zero fill, clamped edges). The resize chain below continues from s1.
    let pick = |len: usize, node: usize| -> usize {
        let mut s1 = node;
        if len >= node {
            while s1 * 2 <= len { s1 *= 2; }
        } else {
            while s1 / 2 >= len { s1 /= 2; }
        }
        s1
    };
    let (s1x, s1y) = (pick(img.w, w), pick(img.h, h));
    let resampled = (s1x, s1y) != (img.w, img.h);
    let img = if !resampled {
        img
    } else {
        let (pw, ph) = (img.w.next_power_of_two(), img.h.next_power_of_two());
        let mut pad = Image::new(pw, ph, img.color, [0.0; 4]);
        for y in 0..img.h {
            for x in 0..img.w {
                pad.px[y * pw + x] = img.px[y * img.w + x];
            }
        }
        let (fx, fy) = (img.w as f32 / s1x as f32, img.h as f32 / s1y as f32);
        let mut o = Image::new(s1x, s1y, img.color, [0.0; 4]);
        for y in 0..s1y {
            for x in 0..s1x {
                o.px[y * s1x + x] = pad.bilinear_or((x as f32 + 0.5) * fx / pw as f32, (y as f32 + 0.5) * fy / ph as f32, (false, false), None, sfmt);
            }
        }
        if !mode.contains("noq1") { o.quantize(fmt); }
        o
    };
    if (img.w, img.h) == (w, h) {
        return img;
    }
    // another size: the identity transformation's resize chain (per axis halvings by pair
    // averages stored in the node's format, then per axis 2x bilinear upsamples)
    let wrap = mode.contains("wrap");
    let upf = sfmt;
    let mut cur = img;
    let mut sample_fmt = if resampled { 3 } else { sfmt };
    let mut halved = false;
    while cur.w > w || cur.h > h {
        halved = true;
        if sample_fmt <= 1 && fmt >= 2 && !mode.contains("nohi") {
            // UNORM file, float node: the first halving averages the 16-bit codes (ties up)
            let (ax, ay) = (cur.w > w, cur.h > h);
            let (nx, ny) = (1 + ax as usize, 1 + ay as usize);
            let mut o = Image::new(cur.w / nx, cur.h / ny, cur.color, [0.0; 4]);
            let n = (nx * ny) as u32;
            for y in 0..o.h {
                for x in 0..o.w {
                    let mut sum = [0u32; 4];
                    for dy in 0..ny { for dx in 0..nx {
                        let p = cur.at(x * nx + dx, y * ny + dy);
                        for c in 0..4 { sum[c] += (p[c].clamp(0.0, 1.0) * 65535.0).round() as u32; }
                    } }
                    o.px[y * o.w + x] = sum.map(|t| ((t + n / 2) / n) as f32 / 65535.0);
                }
            }
            o.quantize(fmt);
            cur = o;
        } else {
            cur = cur.downsample_axes_sampled(cur.w > w, cur.h > h, fmt, sample_fmt);
        }
        sample_fmt = fmt;
    }
    // the first upsample reads the file's own texture (UNORM codes interpolated as integers);
    // the following ones read the float intermediates
    let upq = !mode.contains("noq");
    let mut first = !resampled && !halved;
    while cur.w < w || cur.h < h {
        cur = cur.upsample_axes_fmt(cur.w < w, cur.h < h, (wrap, wrap), if first { upf } else { 3 });
        if upq { cur.quantize(fmt); }
        first = false;
    }
    cur
}

fn from_dynamic(d: &image::DynamicImage, color: bool) -> Image {
    let rgba = d.to_rgba32f();
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    let mut img = Image::new(w, h, color, [0.0; 4]);
    for (k, p) in rgba.pixels().enumerate() {
        img.px[k] = if color { p.0 } else { Image::gray(p.0[0]) };
    }
    img
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as u32;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        f32::from_bits(((e - 15 + 127) as u32) << 23 | m << 13)
    };
    if s == 1 { -v } else { v }
}
