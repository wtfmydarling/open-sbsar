//! The public API: open a .sbsar, inspect its graphs, set parameters, render outputs.
//!
//! ```no_run
//! use open_sbsar::{Package, Parameters, Renderer, Value};
//! let package = Package::open("material.sbsar")?;
//! let graph = &package.graphs()[0];
//! let mut params = Parameters::new();
//! params.set("$outputsize", Value::Int(vec![10, 10])); // log2 sizes: 1024 x 1024
//! params.set("$randomseed", Value::Int(vec![7]));
//! let mut renderer = Renderer::new_gpu().or_else(|_| Ok::<_, open_sbsar::Error>(Renderer::new_cpu()))?;
//! for out in renderer.render(graph, &params, None)? {
//!     println!("{} {}x{} {:?}", out.identifier, out.width, out.height, out.format);
//!     let rgba16: Vec<u16> = out.to_rgba16();
//! }
//! # Ok::<(), open_sbsar::Error>(())
//! ```
use crate::image::Image;
use crate::package::{self, InputInfo, OutputInfo};
use crate::program::Program;
use std::collections::HashMap;

/// Errors of this library.
#[derive(Debug, Clone)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<String> for Error {
    fn from(s: String) -> Self {
        Error(s)
    }
}
impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error(s.into())
    }
}
pub type Result<T> = std::result::Result<T, Error>;

/// A .sbsar archive: one or more graphs.
pub struct Package {
    graphs: Vec<Graph>,
}

impl Package {
    /// Read a .sbsar file (or an .sbsasm with an .xml of the same name next to it).
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Package> {
        let path = path.as_ref();
        if path.extension().is_some_and(|e| e == "sbsasm") {
            let (asm, xml) = package::load_path(path)?;
            return Ok(Package { graphs: vec![Graph::new(path.display().to_string(), asm, xml)?] });
        }
        let bytes = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
    }
    /// Read a .sbsar from memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Package> {
        let graphs = package::read_archive(bytes)?.into_iter().map(|g| Graph::new(g.path, g.asm, g.xml)).collect::<Result<Vec<_>>>()?;
        Ok(Package { graphs })
    }
    pub fn graphs(&self) -> &[Graph] {
        &self.graphs
    }
}

/// One compiled graph: its program, inputs and outputs.
pub struct Graph {
    path: String,
    program: Program,
    xml: String,
    label: Option<String>,
    inputs: Vec<InputInfo>,
    outputs: Vec<OutputInfo>,
    /// unique per graph object (an [`Instance`] belongs to one graph)
    uid: u64,
}

impl Graph {
    fn new(path: String, asm: Vec<u8>, xml: String) -> Result<Graph> {
        let program = Program::parse(asm)?;
        let (label, inputs, mut outputs) = package::metadata(&xml);
        // outputs the program has but the XML does not describe (or no XML at all)
        let names = package::output_names(&xml);
        for o in &program.outputs {
            if !outputs.iter().any(|x| x.uid == o.uid) {
                let identifier = names.iter().find(|(u, _)| *u == o.uid).map_or(o.uid.to_string(), |(_, n)| n.clone());
                outputs.push(OutputInfo { identifier, uid: o.uid, label: None, group: None, usages: vec![], colorspace: None, is_image: true });
            }
        }
        outputs.retain(|x| program.outputs.iter().any(|o| o.uid == x.uid));
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let uid = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Graph { path, program, xml, label, inputs, outputs, uid })
    }
    /// Path of the program inside the archive.
    pub fn path(&self) -> &str {
        &self.path
    }
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
    /// The format version of the compiled program (2..9 are known).
    pub fn format_version(&self) -> u32 {
        self.program.version
    }
    /// Graph inputs (parameters and image inputs), as described by the XML.
    pub fn inputs(&self) -> &[InputInfo] {
        &self.inputs
    }
    /// Graph outputs. `is_image` is false for value outputs (numbers, not textures).
    pub fn outputs(&self) -> &[OutputInfo] {
        &self.outputs
    }
    /// The storage format and channel count an output is computed in.
    pub fn output_format(&self, identifier: &str) -> Option<(PixelFormat, usize)> {
        let info = self.outputs.iter().find(|o| o.identifier == identifier)?;
        let o = self.program.outputs.iter().find(|o| o.uid == info.uid)?;
        let n = self.program.nodes.get(o.node)?;
        Some((PixelFormat::from_storage(n.format()), if n.color { 4 } else { 1 }))
    }
}

/// A parameter value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Float(Vec<f32>),
    Int(Vec<i32>),
    Bool(bool),
    String(String),
}

/// Pixel storage of an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit unsigned normalized
    U8,
    /// 16-bit unsigned normalized
    U16,
    /// IEEE half float
    F16,
    /// IEEE single float
    F32,
}
impl PixelFormat {
    fn from_storage(f: u32) -> PixelFormat {
        match f {
            0 => PixelFormat::U8,
            1 => PixelFormat::U16,
            2 => PixelFormat::F16,
            _ => PixelFormat::F32,
        }
    }
    /// Bytes per channel.
    pub fn bytes(self) -> usize {
        match self {
            PixelFormat::U8 => 1,
            PixelFormat::U16 | PixelFormat::F16 => 2,
            PixelFormat::F32 => 4,
        }
    }
}

/// An image passed to an image input of a graph.
#[derive(Clone, Debug)]
pub struct InputImage {
    pub width: usize,
    pub height: usize,
    /// 1 (gray) or 4 (RGBA)
    pub channels: usize,
    /// pixel values in 0..1 (HDR allowed), row-major, `width * height * channels` values
    pub data: Vec<f32>,
    /// the storage of the source file: `U8` / `U16` values are the integer codes divided by
    /// 255 / 65535 and are interpolated as integers when the image is resized, `F16` / `F32`
    /// as floats
    pub storage: PixelFormat,
}

/// Input values for a render. Inputs that are not set keep the graph's defaults.
#[derive(Clone, Debug, Default)]
pub struct Parameters {
    values: Vec<(String, Value)>,
    images: Vec<(String, InputImage)>,
}
impl Parameters {
    pub fn new() -> Self {
        Self::default()
    }
    /// Set an input by identifier (e.g. `$outputsize`, `$randomseed`, or a graph parameter).
    pub fn set(&mut self, identifier: &str, value: Value) -> &mut Self {
        self.values.retain(|(i, _)| i != identifier);
        self.values.push((identifier.to_string(), value));
        self
    }
    /// Connect an image input (the graph-input nodes read it, resized to their size as
    /// Substance does).
    pub fn set_image(&mut self, identifier: &str, image: InputImage) -> &mut Self {
        self.images.retain(|(i, _)| i != identifier);
        self.images.push((identifier.to_string(), image));
        self
    }
}

/// One rendered output, in the format the graph computes it in.
#[derive(Clone, Debug)]
pub struct OutputImage {
    pub identifier: String,
    pub uid: u32,
    pub width: usize,
    pub height: usize,
    /// 1 (gray) or 4 (RGBA)
    pub channels: usize,
    pub format: PixelFormat,
    /// row-major pixels, `width * height * channels` values of `format`, little endian
    pub data: Vec<u8>,
    /// the same pixels as f32 (exact for every format)
    values: Vec<f32>,
}

impl OutputImage {
    fn new(identifier: String, uid: u32, img: &Image, format: PixelFormat) -> OutputImage {
        let channels = if img.color { 4 } else { 1 };
        let values: Vec<f32> = if img.color { img.px.iter().flatten().copied().collect() } else { img.px.iter().map(|p| p[0]).collect() };
        let mut data = Vec::with_capacity(values.len() * format.bytes());
        for &v in &values {
            match format {
                // stored values are exact multiples of 1/255 and 1/65535
                PixelFormat::U8 => data.push((v.clamp(0.0, 1.0) * 255.0).round() as u8),
                PixelFormat::U16 => data.extend_from_slice(&((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes()),
                PixelFormat::F16 => data.extend_from_slice(&f32_to_f16(v).to_le_bytes()),
                PixelFormat::F32 => data.extend_from_slice(&v.to_le_bytes()),
            }
        }
        OutputImage { identifier, uid, width: img.w, height: img.h, channels, format, data, values }
    }
    /// Pixel values as f32, `width * height * channels`.
    pub fn as_f32(&self) -> &[f32] {
        &self.values
    }
    /// RGBA f32 (gray expanded, alpha 1).
    pub fn to_rgba_f32(&self) -> Vec<f32> {
        if self.channels == 4 { self.values.clone() } else { self.values.iter().flat_map(|&g| [g, g, g, 1.0]).collect() }
    }
    /// RGBA 16-bit, converted as Substance's renderer writes 16-bit images.
    pub fn to_rgba16(&self) -> Vec<u16> {
        self.to_rgba_f32().into_iter().map(|v| crate::image::unorm(v, 16) as u16).collect()
    }
    /// RGBA 8-bit, converted as Substance stores 8-bit images.
    pub fn to_rgba8(&self) -> Vec<u8> {
        self.to_rgba_f32().into_iter().map(|v| crate::image::unorm(v, 8) as u8).collect()
    }
}

/// IEEE half bits of an f32 (round to nearest even; exact for values already stored as half).
fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let r = if rem > mid || (rem == mid && half & 1 == 1) { half + 1 } else { half };
        return sign | r as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    let r = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) { half + 1 } else { half };
    sign | r as u16
}

/// Renders graphs. Keep one renderer and reuse it: the GPU context and its compiled shaders are
/// kept between renders.
pub struct Renderer {
    #[cfg(feature = "gpu")]
    gpu: Option<crate::gpu::Gpu>,
    /// Warnings of the last render (unsupported features, ignored inputs).
    pub warnings: Vec<String>,
    /// identity and render count: an [`Instance`] reuses GPU results only on the renderer, and
    /// the render, that produced them
    id: u64,
    epoch: u64,
}

fn next_renderer_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl Renderer {
    /// The CPU reference path (portable, slower).
    pub fn new_cpu() -> Renderer {
        Renderer {
            #[cfg(feature = "gpu")]
            gpu: None,
            warnings: vec![],
            id: next_renderer_id(),
            epoch: 0,
        }
    }
    /// The GPU path on a hardware Vulkan adapter (feature `gpu`).
    #[cfg(feature = "gpu")]
    pub fn new_gpu() -> Result<Renderer> {
        Ok(Renderer { gpu: Some(crate::gpu::Gpu::new()?), warnings: vec![], id: next_renderer_id(), epoch: 0 })
    }
    /// The GPU path on a device the host application already uses (its `wgpu::Device` and
    /// `wgpu::Queue`, wgpu 30: see [`crate::wgpu`]). Nothing is requested from the device; an
    /// error says which limit is too small. Results are bit-exact with the CPU path on the
    /// hardware this was validated on (see the repository's GPU.md); other drivers may round
    /// differently in the last bits.
    #[cfg(feature = "gpu")]
    pub fn with_device(device: wgpu::Device, queue: wgpu::Queue) -> Result<Renderer> {
        Ok(Renderer { gpu: Some(crate::gpu::Gpu::from_device(device, queue)?), warnings: vec![], id: next_renderer_id(), epoch: 0 })
    }
    /// Whether this renderer uses the GPU.
    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        return self.gpu.is_some();
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// Render `outputs` (identifiers; `None`: every image output) of a graph.
    pub fn render(&mut self, graph: &Graph, params: &Parameters, outputs: Option<&[&str]>) -> Result<Vec<OutputImage>> {
        self.render_impl(graph, params, outputs, None)
    }

    fn render_impl(&mut self, graph: &Graph, params: &Parameters, outputs: Option<&[&str]>, mut inst: Option<&mut Instance>) -> Result<Vec<OutputImage>> {
        self.epoch += 1;
        let wanted: Vec<&OutputInfo> = match outputs {
            None => graph.outputs.iter().filter(|o| o.is_image).collect(),
            Some(ids) => ids
                .iter()
                .map(|id| graph.outputs.iter().find(|o| o.identifier == *id).ok_or_else(|| Error(format!("no output {id}"))))
                .collect::<Result<_>>()?,
        };
        let mut warnings = vec![];
        let mut rt = crate::ops::Runtime::new(&graph.program);
        #[cfg(feature = "gpu")]
        if let Some(g) = self.gpu.take() {
            rt.attach_gpu(g);
        }
        // parameters
        for (uid, s) in package::string_defaults(&graph.xml) {
            rt.strings.insert(uid, s);
        }
        let uids = package::input_uids(&graph.xml);
        for (id, value) in &params.values {
            let Some(uid) = uids.iter().find(|(i, _)| i == id).map(|(_, u)| *u) else {
                warnings.push(format!("no input {id}: ignored"));
                continue;
            };
            let v = match value {
                Value::String(s) => {
                    rt.strings.insert(uid, s.clone());
                    continue;
                }
                Value::Bool(b) => crate::vm::Value::int(&[*b as i32]),
                Value::Int(v) => crate::vm::Value::int(&v[..v.len().min(4)]),
                Value::Float(v) => crate::vm::Value::float(&v[..v.len().min(4)]),
            };
            // an int input given floats (or the reverse) is converted to the input's type
            let v = match (rt.inputs.get(&uid), v) {
                (Some(crate::vm::Value::I(..)), crate::vm::Value::F(n, a)) => crate::vm::Value::I(n, a.map(|x| x as i32)),
                (Some(crate::vm::Value::F(..)), crate::vm::Value::I(n, a)) => crate::vm::Value::F(n, a.map(|x| x as f32)),
                (_, v) => v,
            };
            rt.inputs.insert(uid, v);
        }
        for (id, im) in &params.images {
            let Some(uid) = uids.iter().find(|(i, _)| i == id).map(|(_, u)| *u) else {
                warnings.push(format!("no input {id}: ignored"));
                continue;
            };
            let ch = im.channels;
            if !(ch == 1 || ch == 4) || im.data.len() != im.width * im.height * ch || im.width == 0 || im.height == 0 {
                warnings.push(format!("image input {id}: expected width * height * channels (1 or 4) values: ignored"));
                continue;
            }
            let mut img = crate::image::Image::new(im.width, im.height, ch == 4, [0.0; 4]);
            for (k, q) in img.px.iter_mut().enumerate() {
                *q = if ch == 4 { [im.data[k * 4], im.data[k * 4 + 1], im.data[k * 4 + 2], im.data[k * 4 + 3]] } else { crate::image::Image::gray(im.data[k]) };
            }
            let fmt = match im.storage { PixelFormat::U8 => 0, PixelFormat::U16 => 1, PixelFormat::F16 | PixelFormat::F32 => 3 };
            rt.set_image(uid, img, fmt);
        }
        if let Some(inst) = inst.as_deref_mut() {
            inst.begin(&mut rt, self.id, self.epoch - 1);
        }
        let nodes: Vec<(usize, &OutputInfo)> = wanted
            .iter()
            .filter_map(|o| graph.program.outputs.iter().find(|p| p.uid == o.uid).map(|p| (p.node, *o)))
            .collect();
        let targets: Vec<usize> = nodes.iter().map(|(k, _)| *k).collect();
        // the engine reports internal failures by panicking: turn them into errors
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rt.run_for(&targets);
            let mut out = Vec::new();
            for (k, o) in &nodes {
                let img = rt.images[*k].as_ref().ok_or_else(|| Error(format!("output {} not computed", o.identifier)))?;
                let format = PixelFormat::from_storage(graph.program.nodes[*k].format());
                out.push(OutputImage::new(o.identifier.clone(), o.uid, img, format));
            }
            rt.incr_finish();
            Ok::<_, Error>(out)
        }));
        for (_, why) in &rt.unsupported {
            if !warnings.iter().any(|w| w.ends_with(why.as_str())) {
                warnings.push(format!("unsupported: {why}"));
            }
        }
        let ok = matches!(run, Ok(Ok(_)));
        #[cfg(feature = "gpu")]
        {
            self.gpu = if inst.is_some() && ok { rt.take_gpu() } else { rt.detach_gpu() };
        }
        if let Some(inst) = inst {
            inst.end(&mut rt, ok, self.id, self.epoch);
        }
        self.warnings = warnings;
        match run {
            Ok(r) => r,
            Err(p) => {
                let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                #[cfg(feature = "gpu")]
                {
                    // a GPU context interrupted mid-render is not reused
                    self.gpu = None;
                }
                Err(Error(format!("render failed: {msg}")))
            }
        }
    }
}

/// Counts of the last [`Instance::render`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderStats {
    /// nodes computed in the last render
    pub recomputed: usize,
    /// nodes whose kept result was reused
    pub reused: usize,
}

/// A graph being edited: renders it again after parameter changes, recomputing only the nodes
/// whose result can change and reusing the kept results of the others. The outputs are
/// identical to [`Renderer::render`] with the same parameters.
///
/// A node is reused when none of the graph inputs its functions read changed, none of its
/// input nodes was recomputed, and the value-processor values it read are the same. Results
/// are kept within a memory budget (default 512 MiB; [`Instance::set_cache_budget`]); what does
/// not fit is recomputed when needed. Use one instance with one renderer: rendering with
/// another renderer, or rendering anything else on the same renderer in between, makes the
/// next render a full one. GPU results stay on the renderer's device.
///
/// ```no_run
/// use open_sbsar::{Instance, Package, Parameters, Renderer, Value};
/// let package = Package::open("material.sbsar")?;
/// let graph = &package.graphs()[0];
/// let mut renderer = Renderer::new_gpu()?;
/// let mut instance = Instance::new(graph);
/// let mut params = Parameters::new();
/// params.set("$outputsize", Value::Int(vec![9, 9]));
/// let first = instance.render(&mut renderer, graph, &params, None)?;   // full render
/// params.set("$randomseed", Value::Int(vec![3]));
/// let again = instance.render(&mut renderer, graph, &params, None)?;   // only what changed
/// println!("{:?}", instance.last_stats());
/// # Ok::<(), open_sbsar::Error>(())
/// ```
pub struct Instance {
    graph: u64,
    /// (renderer id, its render count after our last render): the kept results are valid
    renderer: Option<(u64, u64)>,
    images: Vec<Option<std::sync::Arc<Image>>>,
    incr: Option<crate::ops::Incr>,
    inputs: HashMap<u32, crate::vm::Value>,
    strings: HashMap<u32, String>,
    image_inputs: HashMap<u32, (std::sync::Arc<Image>, u32)>,
    budget: usize,
    stats: RenderStats,
}

impl Instance {
    /// An instance of `graph` with an empty cache.
    pub fn new(graph: &Graph) -> Instance {
        Instance {
            graph: graph.uid,
            renderer: None,
            images: vec![],
            incr: None,
            inputs: HashMap::new(),
            strings: HashMap::new(),
            image_inputs: HashMap::new(),
            budget: 512 << 20,
            stats: RenderStats::default(),
        }
    }
    /// Render `outputs` (`None`: every image output) of `graph` (the graph this instance was
    /// created for) with `params`, reusing what the previous render computed.
    pub fn render(&mut self, renderer: &mut Renderer, graph: &Graph, params: &Parameters, outputs: Option<&[&str]>) -> Result<Vec<OutputImage>> {
        if graph.uid != self.graph {
            return Err(Error("Instance::render: another graph than the instance's".into()));
        }
        renderer.render_impl(graph, params, outputs, Some(self))
    }
    /// Nodes recomputed and reused by the last render.
    pub fn last_stats(&self) -> RenderStats {
        self.stats
    }
    /// Drop the kept results: the next render is a full one.
    pub fn clear(&mut self) {
        self.renderer = None;
        self.images.clear();
        self.incr = None;
    }
    /// Memory for kept results, in bytes (16 per pixel). Lowering it clears the cache.
    pub fn set_cache_budget(&mut self, bytes: usize) {
        if bytes < self.budget {
            self.clear();
        }
        self.budget = bytes;
    }

    fn begin(&mut self, rt: &mut crate::ops::Runtime, renderer: u64, epoch: u64) {
        let n = rt.p.nodes.len();
        let valid = self.renderer == Some((renderer, epoch)) && self.images.len() == n && self.incr.is_some();
        let mut incr = if valid { self.incr.take().unwrap() } else { crate::ops::Incr::default() };
        incr.budget = self.budget;
        incr.recomputed = 0;
        incr.reused = 0;
        incr.changed.clear();
        if valid {
            let mut changed = std::collections::HashSet::new();
            for (u, v) in rt.inputs.iter() {
                if self.inputs.get(u) != Some(v) { changed.insert(*u); }
            }
            for u in self.inputs.keys() {
                if !rt.inputs.contains_key(u) { changed.insert(*u); }
            }
            for (u, v) in rt.strings.iter() {
                if self.strings.get(u) != Some(v) { changed.insert(*u); }
            }
            for u in self.strings.keys() {
                if !rt.strings.contains_key(u) { changed.insert(*u); }
            }
            let same = |a: &(std::sync::Arc<Image>, u32), b: &(std::sync::Arc<Image>, u32)| {
                a.1 == b.1 && a.0.w == b.0.w && a.0.h == b.0.h && a.0.color == b.0.color
                    && a.0.px.iter().zip(&b.0.px).all(|(p, q)| p.iter().zip(q).all(|(x, y)| x.to_bits() == y.to_bits()))
            };
            for (u, v) in rt.image_inputs.iter() {
                if !self.image_inputs.get(u).is_some_and(|w| same(v, w)) { changed.insert(*u); }
            }
            for u in self.image_inputs.keys() {
                if !rt.image_inputs.contains_key(u) { changed.insert(*u); }
            }
            incr.changed = changed;
            rt.images = std::mem::take(&mut self.images);
        } else {
            self.images.clear();
        }
        self.inputs = rt.inputs.clone();
        self.strings = rt.strings.clone();
        self.image_inputs = rt.image_inputs.clone();
        rt.incr = Some(incr);
    }

    fn end(&mut self, rt: &mut crate::ops::Runtime, ok: bool, renderer: u64, epoch: u64) {
        let incr = rt.incr.take();
        self.stats = incr.as_ref().map_or_else(RenderStats::default, |c| RenderStats { recomputed: c.recomputed, reused: c.reused });
        if ok {
            self.images = std::mem::take(&mut rt.images);
            self.incr = incr;
            self.renderer = Some((renderer, epoch));
        } else {
            self.clear();
        }
    }
}

#[allow(dead_code)]
fn _assert_send() {
    fn is_send<T: Send>() {}
    is_send::<Package>();
    is_send::<Parameters>();
    is_send::<OutputImage>();
    let _ = HashMap::<u32, u32>::new();
}
