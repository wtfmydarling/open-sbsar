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

/// One pass of the transformation kernel: its Config words (`header.x` selects sample /
/// permute / halve / upsample, see gpu_xf.wgsl), output size, colour flag and storage format.
pub struct XfPass {
    pub config: [u32; 40],
    pub w: usize,
    pub h: usize,
    pub color: bool,
    pub format: u32,
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    xf_pipeline: wgpu::ComputePipeline,
    layout: wgpu::PipelineLayout,
    pp: HashMap<String, wgpu::ComputePipeline>,
    pub dispatched: usize,
    pub downloads: usize,
    pub uploads: usize,
    pub input_copies: usize,
    images: HashMap<usize, Resident>,
}
impl Gpu {
    pub fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or("no Vulkan adapter")?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err("Vulkan selected a software CPU adapter; hardware GPU required".into());
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor { label: Some("sdrt"), required_features: wgpu::Features::empty(), required_limits: wgpu::Limits::default() },
            None,
        ))
        .map_err(|e| e.to_string())?;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdrt basic atomic nodes"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", include_str!("gpu_prelude.wgsl"), include_str!("gpu_atomic.wgsl")).into(),
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
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdrt atomic"),
            layout: Some(&layout),
            module: &module,
            entry_point: "main",
        });
        // The transformation kernel shares the prelude (config, quantize, store) and the
        // same bind group layout, but samples its source instead of combining two inputs.
        let xf_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdrt transformation"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", include_str!("gpu_prelude.wgsl"), include_str!("gpu_xf.wgsl")).into(),
            ),
        });
        let xf_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdrt transformation"),
            layout: Some(&layout),
            module: &xf_module,
            entry_point: "main",
        });
        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            return Err(error.to_string());
        }
        eprintln!("GPU: {} ({:?}, {}); initial hybrid backend", info.name, info.backend, info.driver);
        Ok(Self { device, queue, pipeline, xf_pipeline, layout, pp: HashMap::new(), dispatched: 0, downloads: 0, uploads: 0, input_copies: 0, images: HashMap::new() })
    }

    pub fn size(&self, id: usize) -> Option<(usize, usize)> {
        self.images.get(&id).map(|i| (i.w, i.h))
    }
    pub fn release(&mut self, id: usize) {
        self.images.remove(&id);
    }
    pub fn reset(&mut self) {
        self.images.clear();
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
        shader: Option<&str>,
        pp_inputs: &[(Option<usize>,u32)],
    ) -> Result<(), String> {
        if let Some(source) = shader {
            if !self.pp.contains_key(source) {
                let start = std::time::Instant::now();
                self.device.push_error_scope(wgpu::ErrorFilter::Validation);
                let module =
                    self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("PP"), source: wgpu::ShaderSource::Wgsl(source.into()) });
                let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("PP"),
                    layout: Some(&self.layout),
                    module: &module,
                    entry_point: "main",
                });
                if let Some(e) = pollster::block_on(self.device.pop_error_scope()) {
                    return Err(e.to_string());
                }
                if std::env::var("SDRT_GPU_LOG").is_ok() {
                    eprintln!("GPU: PP pipeline for node {id} created in {:.1} ms", start.elapsed().as_secs_f64() * 1e3);
                }
                self.pp.insert(source.into(), pipeline);
            }
        }
        let pipeline = shader.map_or(&self.pipeline, |s| &self.pp[s]);
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
        let upload = |label, data: &[u8]| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: data, usage: wgpu::BufferUsages::STORAGE })
        };
        let mut config = *config;
        config[1] = count as u32;
        config[2] = w as u32;
        config[16] = format;
        config[17] = u32::from(color);
        config[18]=u32::from(pp_inputs.len()>2);
        config[19]=pp_inputs.len() as u32;
        let params = upload("params", bytemuck::cast_slice(&config));
        let zero = [[0f32; 4]];
        let dummy = Arc::new(upload("unused input", bytemuck::cast_slice(&zero)));
        let mut ba = a.map_or_else(|| dummy.clone(), |id| self.images[&id].buffer.clone());
        let mut bb = b.map_or_else(|| dummy.clone(), |id| self.images[&id].buffer.clone());
        let mut encoder=self.device.create_command_encoder(&Default::default());
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
            ba=Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {label:Some("packed PP inputs"),size:total.max(16),mapped_at_creation:false,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST}));
            for (id,offset) in copies {
                let im=&self.images[&id];
                encoder.copy_buffer_to_buffer(&im.buffer,0,&ba,offset,(im.w*im.h*16) as u64);
                self.input_copies+=1;
            }
            bb=Arc::new(upload("PP input descriptors",bytemuck::cast_slice(&descriptors)));
        }
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: bytes,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let buffers = [&params, ba.as_ref(), bb.as_ref(), &out];
        let entries: Vec<_> = buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("atomic inputs"), layout: &layout, entries: &entries });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("atomic"), timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.images.insert(id, Resident { buffer: Arc::new(out), w, h, color });
        self.dispatched += 1;
        // Reclaim completed submissions without waiting for a readback boundary.
        if self.dispatched % 32 == 0 {
            self.device.poll(wgpu::Maintain::Poll);
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

    /// The transformation kernel (op 0x04, gpu_xf.wgsl): a chain of passes recorded in one
    /// submission. The first pass reads the resident `source`, each later pass the previous
    /// pass's output; the last output becomes resident image `id`. Intermediate buffers
    /// (mip levels, per-axis halvings and 2x upsamples) are not graph nodes and are freed
    /// once the GPU is done with them. Each pass's `selectors` must describe its input.
    pub fn dispatch_transform(&mut self, id: usize, source: usize, passes: &[XfPass]) -> Result<(), String> {
        let src = self.images.get(&source).ok_or("transform source is not resident on the GPU")?;
        let (mut input, mut iw, mut ih) = (src.buffer.clone(), src.w, src.h);
        let last = passes.last().ok_or("transform without passes")?;
        let limits = self.device.limits();
        let layout = self.xf_pipeline.get_bind_group_layout(0);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for pass in passes {
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
            if (iw as u32, ih as u32) != (pass.config[12], pass.config[13]) {
                return Err(format!("transform pass input descriptor {}x{} does not match its {iw}x{ih} input", pass.config[12], pass.config[13]));
            }
            let mut config = pass.config;
            config[1] = count as u32;
            config[2] = w as u32;
            config[16] = pass.format;
            config[17] = u32::from(pass.color);
            let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("transform params"),
                contents: bytemuck::cast_slice(&config),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let out = Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("transform output"),
                size: bytes,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }));
            // Binding 2 is unused by the kernel; the input is bound there too (both read-only).
            let entries: Vec<_> = [&params, input.as_ref(), input.as_ref(), out.as_ref()]
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                .collect();
            let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("transform inputs"), layout: &layout, entries: &entries });
            // One compute pass per step: wgpu orders the write before the next pass's read.
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("transform"), timestamp_writes: None });
                cpass.set_pipeline(&self.xf_pipeline);
                cpass.set_bind_group(0, &bind, &[]);
                cpass.dispatch_workgroups(groups[0], groups[1], 1);
            }
            (input, iw, ih) = (out, w, h);
            self.dispatched += 1;
        }
        self.queue.submit(Some(encoder.finish()));
        self.images.insert(id, Resident { buffer: input, w: last.w, h: last.h, color: last.color });
        self.device.poll(wgpu::Maintain::Poll);
        Ok(())
    }
    pub fn download(&mut self, id: usize) -> Result<Image, String> {
        let im = self.images.get(&id).ok_or("missing GPU image")?;
        let (w, h, color) = (im.w, im.h, im.color);
        let bytes = (w * h * 16) as u64;
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CPU boundary readback"),
            size: bytes,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&im.buffer, 0, &read, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let mapped = slice.get_mapped_range();
        let px = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped).to_vec();
        drop(mapped);
        read.unmap();
        self.downloads += 1;
        Ok(Image { w, h, color, px })
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        eprintln!("GPU: {} cached PP pipelines, {} device input copies", self.pp.len(), self.input_copies);
        eprintln!("GPU: {} compute dispatches, {} uploads, {} readbacks (other nodes ran on CPU)", self.dispatched, self.uploads, self.downloads);
    }
}
