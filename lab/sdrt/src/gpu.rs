//! Hybrid Vulkan backend with resident graph buffers and explicit CPU boundaries.
use crate::image::Image;
use std::{collections::HashMap, sync::Arc};
use wgpu::util::DeviceExt;
struct Resident {
    buffer: Arc<wgpu::Buffer>,
    w: usize,
    h: usize,
    color: bool,
}

/// Where a pass reads `input_a` / `input_b` (bindings 1 and 2) from.
#[derive(Clone, Copy, Debug)]
pub enum Src {
    /// a one-texel dummy buffer
    None,
    /// a resident graph image
    Node(usize),
    /// the output of an earlier pass of the same chain
    Pass(usize),
}

/// One pass of a kernel chain (see `Gpu::dispatch_passes`): its Config words, output size,
/// colour flag and storage format (the prelude's `store` quantizes to them), and its inputs.
pub struct Pass {
    pub config: [u32; 40],
    pub w: usize,
    pub h: usize,
    pub color: bool,
    pub format: u32,
    pub a: Src,
    pub b: Src,
}

/// The transformation kernel (gpu_xf.wgsl): `header.x` selects sample / permute / halve /
/// upsample.
pub const XF: (&str, &str) = ("transformation", include_str!("gpu_xf.wgsl"));
pub const ATOMIC: (&str, &str) = ("atomic", concat!(include_str!("gpu_sample.wgsl"), include_str!("gpu_atomic.wgsl")));

/// Upper bound of the bytes of released image buffers kept for reuse.
const POOL_BYTES: u64 = 512 << 20;

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: Arc<wgpu::ComputePipeline>,
    /// prelude + one WGSL file each, compiled on first use (see `dispatch_passes`)
    kernels: HashMap<&'static str, Arc<wgpu::ComputePipeline>>,
    layout: wgpu::PipelineLayout,
    pp: HashMap<String, Arc<wgpu::ComputePipeline>>,
    pub dispatched: usize,
    pub downloads: usize,
    pub uploads: usize,
    pub input_copies: usize,
    images: HashMap<usize, Resident>,
    dummy: Arc<wgpu::Buffer>,
    encoder: Option<wgpu::CommandEncoder>,
    pending_dispatches: usize,
    pool: HashMap<u64, Vec<Arc<wgpu::Buffer>>>,
    pending_buffers: Vec<Arc<wgpu::Buffer>>,
    read_pool: HashMap<u64, Vec<wgpu::Buffer>>,
    /// bytes of the buffers in `pool` (bounded by POOL_BYTES)
    pooled: u64,
    /// bytes of the buffers in `read_pool` (bounded by POOL_BYTES)
    read_pooled: u64,
    params_pending: Vec<wgpu::Buffer>,
}
impl Gpu {
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..wgpu::InstanceDescriptor::new_without_display_handle() });
        let (info, device, queue) = {
            let mut last_err = String::new();
            let mut res = None;
            for _ in 0..100 {
                let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
        apply_limit_buckets: false,
                })) else {
                    last_err = "no Vulkan adapter".into();
                    std::thread::sleep(std::time::Duration::from_millis(250));
                    continue;
                };
                let info = adapter.get_info();
                if info.device_type == wgpu::DeviceType::Cpu {
                    return Err("Vulkan selected a software CPU adapter; hardware GPU required".into());
                }
                let mut required_limits = wgpu::Limits::default();
                let ad_limits = adapter.limits();
                required_limits.max_storage_buffer_binding_size = ad_limits.max_storage_buffer_binding_size;
                required_limits.max_buffer_size = ad_limits.max_buffer_size;
                let dq = pollster::block_on(adapter.request_device(
                    &wgpu::DeviceDescriptor { label: Some("sdrt"), required_features: wgpu::Features::empty(), required_limits, ..Default::default() },
                ));
                drop(adapter);
                match dq {
                    Ok((device, queue)) => { res = Some((info, device, queue)); break; }
                    Err(e) => {
                        last_err = e.to_string();
                        std::thread::sleep(std::time::Duration::from_millis(250));
                    }
                }
            }
            res.ok_or(last_err)?
        };
        if std::env::var_os("SDRT_GPU_LOG").is_some() {
            eprintln!("GPU: {} ({:?}, {}); initial hybrid backend", info.name, info.backend, info.driver);
        }
        Self::from_device(device, queue)
    }

    pub fn from_device(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, String> {
        let limits = device.limits();
        if limits.max_storage_buffer_binding_size < 160 || limits.max_buffer_size < 160
            || limits.max_storage_buffers_per_shader_stage < 4
            || limits.max_bind_groups < 1 || limits.max_bindings_per_bind_group < 4
            || limits.max_compute_workgroup_size_x < 8 || limits.max_compute_workgroup_size_y < 8
            || limits.max_compute_invocations_per_workgroup < 64 || limits.max_compute_workgroups_per_dimension < 1 {
            return Err("GPU device limits are too small: need four storage buffers, 160-byte bindings and 8x8 compute workgroups".into());
        }
        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdrt basic atomic nodes"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}\n{}", include_str!("gpu_prelude.wgsl"), include_str!("gpu_sample.wgsl"), include_str!("gpu_atomic.wgsl")).into(),
            ),
        });
        let entries: Vec<_> = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: binding != 3 },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("node buffers"), entries: &entries });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("node layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdrt atomic"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(error.to_string());
        }
        let dummy = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shared dummy"),
            contents: bytemuck::cast_slice(&[[0f32; 4]]),
            usage: wgpu::BufferUsages::STORAGE,
        }));
        Ok(Self {
            device,
            queue,
            pipeline: Arc::new(pipeline),
            kernels: HashMap::new(),
            layout,
            pp: HashMap::new(),
            dispatched: 0,
            downloads: 0,
            uploads: 0,
            input_copies: 0,
            images: HashMap::new(),
            dummy,
            encoder: None,
            pending_dispatches: 0,
            pool: HashMap::new(),
            pending_buffers: Vec::new(),
            read_pool: HashMap::new(),
            pooled: 0,
            read_pooled: 0,
            params_pending: Vec::new(),
        })
    }

    fn ensure_encoder(&mut self) -> &mut wgpu::CommandEncoder {
        if self.encoder.is_none() {
            self.encoder = Some(self.device.create_command_encoder(&Default::default()));
        }
        self.encoder.as_mut().unwrap()
    }

    pub fn flush(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            self.queue.submit(Some(encoder.finish()));
            self.pending_dispatches = 0;
            self.reclaim();
        }
    }

    fn reclaim(&mut self) {
        // released image buffers are kept for reuse up to POOL_BYTES, the rest are freed (wgpu
        // frees a dropped buffer once the submitted work using it is done)
        for buf in std::mem::take(&mut self.pending_buffers) {
            if self.pooled + buf.size() <= POOL_BYTES {
                self.pooled += buf.size();
                self.pool.entry(buf.size()).or_default().push(buf);
            }
        }
        self.params_pending.clear();
    }

    fn get_buffer(&mut self, size: u64, label: &'static str) -> Arc<wgpu::Buffer> {
        if let Some(list) = self.pool.get_mut(&size) {
            while let Some(buf) = list.pop() {
                self.pooled -= size;
                if Arc::strong_count(&buf) == 1 {
                    return buf;
                }
            }
        }
        Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        }))
    }

    fn get_params_buffer(&mut self, words: &[u32]) -> (wgpu::Buffer, bool) {
        (self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::STORAGE,
        }), false)
    }

    /// Free every pooled buffer (GPU memory is given back to the device).
    pub fn trim(&mut self) {
        self.flush();
        self.pool.clear();
        self.read_pool.clear();
        self.pooled = 0;
        self.read_pooled = 0;
    }

    fn get_read_buffer(&mut self, bytes: u64) -> wgpu::Buffer {
        if let Some(list) = self.read_pool.get_mut(&bytes) {
            if let Some(buf) = list.pop() {
                self.read_pooled -= bytes;
                return buf;
            }
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CPU boundary readback"),
            size: bytes,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn recycle_read_buffer(&mut self, bytes: u64, buf: wgpu::Buffer) {
        if self.read_pooled + bytes <= POOL_BYTES {
            self.read_pooled += bytes;
            self.read_pool.entry(bytes).or_default().push(buf);
        }
    }

    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    pub fn size(&self, id: usize) -> Option<(usize, usize)> {
        self.images.get(&id).map(|i| (i.w, i.h))
    }
    pub fn release(&mut self, id: usize) {
        if let Some(im) = self.images.remove(&id) {
            self.pending_buffers.push(im.buffer);
        }
    }
    pub fn reset(&mut self) {
        self.flush();
        for (_, im) in self.images.drain() {
            self.pending_buffers.push(im.buffer);
        }
        self.reclaim();
    }
    pub fn upload(&mut self, id: usize, im: &Image) -> Result<(), String> {
        if self.images.contains_key(&id) {
            return Ok(());
        }
        let bytes = (im.px.len() as u64) * 16;
        if bytes == 0 || bytes > self.device.limits().max_storage_buffer_binding_size as u64 || bytes > self.device.limits().max_buffer_size {
            return Err(format!("GPU input buffer limits exceeded for {}x{}", im.w, im.h));
        }
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CPU boundary upload"),
            contents: bytemuck::cast_slice(&im.px),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        self.images.insert(id, Resident { buffer: Arc::new(buffer), w: im.w, h: im.h, color: im.color });
        self.uploads += 1;
        Ok(())
    }
    pub fn dispatch(
        &mut self,
        id: usize,
        w: usize,
        h: usize,
        color: bool,
        format: u32,
        config: &[u32; 40],
        a: Option<usize>,
        b: Option<usize>,
        shader: Option<(&str, &[u32])>,
        pp_inputs: &[(Option<usize>,u32)],
    ) -> Result<(), String> {
        if let Some((source, _)) = shader {
            if !self.pp.contains_key(source) {
                let start = std::time::Instant::now();
                let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
                let module =
                    self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("PP"), source: wgpu::ShaderSource::Wgsl(source.into()) });
                let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("PP"),
                    layout: Some(&self.layout),
                    module: &module,
                    entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
                });
                if let Some(e) = pollster::block_on(error_scope.pop()) {
                    return Err(e.to_string());
                }
                if std::env::var("SDRT_GPU_LOG").is_ok() {
                    eprintln!("GPU: PP pipeline for node {id} created in {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
                }
                self.pp.insert(source.into(), Arc::new(pipeline));
            }
        }
        let pipeline = shader.map_or_else(|| self.pipeline.clone(), |(s, _)| self.pp[s].clone());
        let count = w.checked_mul(h).ok_or("GPU image size overflow")?;
        let bytes = (count as u64).checked_mul(16).ok_or("GPU buffer size overflow")?;
        let groups = [(w as u32).div_ceil(8), (h as u32).div_ceil(8)];
        let limits = self.device.limits();
        if bytes == 0
            || bytes > limits.max_storage_buffer_binding_size as u64
            || bytes > limits.max_buffer_size
            || groups.iter().any(|&g| g > limits.max_compute_workgroups_per_dimension)
        {
            return Err(format!("GPU buffer limits exceeded for {w}x{h}; no silent CPU fallback"));
        }
        let mut config = *config;
        config[1] = count as u32;
        config[2] = w as u32;
        config[16] = format;
        config[17] = u32::from(color);
        config[18]=u32::from(pp_inputs.len()>2);
        config[19]=pp_inputs.len() as u32;
        // A PP shader's per-node values follow the 40 Config words (its `config.lits`): one
        // word at least, padded to the struct's 16-byte alignment (the binding size the
        // shader expects).
        let mut words = config.to_vec();
        if let Some((_, lits)) = shader {
            words.extend_from_slice(lits);
            words.resize((words.len() + 1).next_multiple_of(4), 0);
        }
        let (params, params_reusable) = self.get_params_buffer(&words);
        let mut ba = a.map_or_else(|| self.dummy.clone(), |id| self.images[&id].buffer.clone());
        let mut bb = b.map_or_else(|| self.dummy.clone(), |id| self.images[&id].buffer.clone());
        let mut packed_ba: Option<Arc<wgpu::Buffer>> = None;
        let mut packed_bb: Option<Arc<wgpu::Buffer>> = None;
        if pp_inputs.len()>2 {
            // Portable variable input count: pack distinct images with GPU-to-GPU copies.
            // Repeated slots share the same offset; no CPU image readback is involved.
            let mut offsets=HashMap::new(); let mut copies=Vec::new(); let mut total=0u64;
            let mut descriptors=Vec::<[u32;4]>::new();
            for &(id,fmt) in pp_inputs {
                if let Some(id)=id {
                    let im=&self.images[&id];
                    let offset=*offsets.entry(id).or_insert_with(|| {
                        let offset=total;total+=(im.w*im.h*16) as u64;copies.push((id,offset));offset
                    });
                    descriptors.push([im.w as u32,im.h as u32,fmt,u32::from(im.color)]);
                    descriptors.push([(offset/16) as u32,0,0,0]);
                } else {descriptors.extend([[0;4];2]);}
            }
            if total.max(16)>limits.max_storage_buffer_binding_size as u64 || total.max(16)>limits.max_buffer_size || (descriptors.len()*16) as u64>limits.max_storage_buffer_binding_size as u64 {
                return Err("packed PP inputs exceed GPU binding size limit".into());
            }
            let ba_buf = self.get_buffer(total.max(16), "packed PP inputs");
            let copy_specs: Vec<(Arc<wgpu::Buffer>, u64, u64)> = copies
                .into_iter()
                .map(|(id, offset)| {
                    let im = &self.images[&id];
                    (im.buffer.clone(), offset, (im.w * im.h * 16) as u64)
                })
                .collect();
            self.input_copies += copy_specs.len();
            let encoder = self.ensure_encoder();
            for (buf, offset, sz) in copy_specs {
                encoder.copy_buffer_to_buffer(&buf, 0, &ba_buf, offset, sz);
            }
            ba = ba_buf.clone();
            packed_ba = Some(ba_buf);
            let bb_buf = Arc::new(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PP input descriptors"),
                contents: bytemuck::cast_slice(&descriptors),
                // returned to the buffer pool afterwards and reused as an image buffer: it needs
                // the pooled buffers' usages (a STORAGE-only buffer broke a later readback copy)
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }));
            bb = bb_buf.clone();
            packed_bb = Some(bb_buf);
        }
        let out = self.get_buffer(bytes, "output");
        let layout = pipeline.get_bind_group_layout(0);
        let buffers = [&params, ba.as_ref(), bb.as_ref(), out.as_ref()];
        let entries: Vec<_> = buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("atomic inputs"), layout: &layout, entries: &entries });
        {
            let encoder = self.ensure_encoder();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("atomic"), timestamp_writes: None });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], 1);
        }
        if params_reusable {
            self.params_pending.push(params);
        }
        if let Some(pba) = packed_ba { self.pending_buffers.push(pba); }
        if let Some(pbb) = packed_bb { self.pending_buffers.push(pbb); }

        self.images.insert(id, Resident { buffer: out, w, h, color });
        self.dispatched += 1;
        self.pending_dispatches += 1;
        if self.pending_dispatches >= 64 {
            self.flush();
        }
        Ok(())
    }
    /// Whether a w x h RGBA f32 image fits one storage binding.
    pub fn fits(&self, w: usize, h: usize) -> bool {
        let limits = self.device.limits();
        let bytes = (w as u64).saturating_mul(h as u64).saturating_mul(16);
        bytes > 0 && bytes <= limits.max_storage_buffer_binding_size as u64 && bytes <= limits.max_buffer_size
    }
    pub fn color(&self, id: usize) -> Option<bool> {
        self.images.get(&id).map(|i| i.color)
    }

    /// A chain of passes of one kernel (`(name, WGSL)`, appended to the prelude and compiled
    /// on first use), recorded in one submission. Each pass reads graph images or earlier
    /// passes' outputs; the last output becomes resident image `id`. Intermediate buffers are
    /// not graph nodes and are freed once the GPU is done with them.
    pub fn dispatch_passes(&mut self, id: usize, kernel: (&'static str, &'static str), passes: &[Pass]) -> Result<(), String> {
        let last = passes.last().ok_or("kernel chain without passes")?;
        if !self.kernels.contains_key(kernel.0) {
            let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(kernel.0),
                source: wgpu::ShaderSource::Wgsl(format!("{}\n{}", include_str!("gpu_prelude.wgsl"), kernel.1).into()),
            });
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(kernel.0),
                layout: Some(&self.layout),
                module: &module,
                entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
            });
            if let Some(e) = pollster::block_on(error_scope.pop()) {
                return Err(format!("{} kernel: {e}", kernel.0));
            }
            self.kernels.insert(kernel.0, Arc::new(pipeline));
        }
        let limits = self.device.limits();
        let mut outputs: Vec<Arc<wgpu::Buffer>> = Vec::with_capacity(passes.len());
        for (i, pass) in passes.iter().enumerate() {
            let (w, h) = (pass.w, pass.h);
            let count = w.checked_mul(h).ok_or("GPU image size overflow")?;
            let bytes = (count as u64).checked_mul(16).ok_or("GPU buffer size overflow")?;
            let groups = [(w as u32).div_ceil(8), (h as u32).div_ceil(8)];
            if bytes == 0
                || bytes > limits.max_storage_buffer_binding_size as u64
                || bytes > limits.max_buffer_size
                || groups.iter().any(|&g| g > limits.max_compute_workgroups_per_dimension)
            {
                return Err(format!("GPU buffer limits exceeded for {w}x{h}; no silent CPU fallback"));
            }
            let source = |s: Src, dummy: &Arc<wgpu::Buffer>, outputs: &[Arc<wgpu::Buffer>]| -> Result<Arc<wgpu::Buffer>, String> {
                match s {
                    Src::None => Ok(dummy.clone()),
                    Src::Node(k) => self.images.get(&k).map(|r| r.buffer.clone()).ok_or_else(|| format!("{} pass {i}: node {k} is not resident", kernel.0)),
                    Src::Pass(j) if j < i => Ok(outputs[j].clone()),
                    Src::Pass(j) => Err(format!("{} pass {i} reads pass {j}", kernel.0)),
                }
            };
            let a = source(pass.a, &self.dummy, &outputs)?;
            let b = source(pass.b, &self.dummy, &outputs)?;
            let mut config = pass.config;
            config[1] = count as u32;
            config[2] = w as u32;
            config[16] = pass.format;
            config[17] = u32::from(pass.color);
            let (params, params_reusable) = self.get_params_buffer(&config);
            let out = self.get_buffer(bytes, "pass output");
            let pipeline = self.kernels[kernel.0].clone();
            let layout = pipeline.get_bind_group_layout(0);
            let entries: Vec<_> = [&params, a.as_ref(), b.as_ref(), out.as_ref()]
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                .collect();
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("pass inputs"), layout: &layout, entries: &entries });
            // One compute pass per step: wgpu orders each write before the next pass's read.
            {
                let encoder = self.ensure_encoder();
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(kernel.0), timestamp_writes: None });
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind, &[]);
                cpass.dispatch_workgroups(groups[0], groups[1], 1);
            }
            if params_reusable {
                self.params_pending.push(params);
            }
            outputs.push(out);
            self.dispatched += 1;
            self.pending_dispatches += 1;
        }
        let last_out = outputs.pop().unwrap();
        for buf in outputs {
            self.pending_buffers.push(buf);
        }
        self.images.insert(id, Resident { buffer: last_out, w: last.w, h: last.h, color: last.color });
        if self.pending_dispatches >= 64 {
            self.flush();
        }
        Ok(())
    }
    pub fn download(&mut self, id: usize) -> Result<Image, String> {
        let im = self.images.get(&id).ok_or("missing GPU image")?;
        let (w, h, color, im_buf) = (im.w, im.h, im.color, im.buffer.clone());
        let bytes = (w * h * 16) as u64;
        let read = self.get_read_buffer(bytes);
        self.ensure_encoder().copy_buffer_to_buffer(&im_buf, 0, &read, 0, bytes);
        self.flush();
        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let mapped = slice.get_mapped_range().expect("mapped buffer range");
        let px = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped).to_vec();
        drop(mapped);
        read.unmap();
        self.recycle_read_buffer(bytes, read);
        self.downloads += 1;
        Ok(Image { w, h, color, px })
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        self.flush();
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if std::env::var_os("SDRT_GPU_LOG").is_some() {
        eprintln!("GPU: {} cached PP pipelines, {} device input copies", self.pp.len(), self.input_copies);
        eprintln!("GPU: {} compute dispatches, {} uploads, {} readbacks (other nodes ran on CPU)", self.dispatched, self.uploads, self.downloads);
        }
    }
}
