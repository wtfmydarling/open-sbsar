//! A research probe: run a WGSL snippet on the GPU with a source image bound, and print
//! the values it produced.
//!
//! It answers one question the runtime cannot answer by itself: how the driver's shader
//! compiler rounds a given sequence of floating-point operations, so that the reference
//! CPU formulas can be matched exactly in the GPU kernels.
//!
//! Usage: gpu_fma <wgsl file> <source .f32>

use wgpu::util::DeviceExt;
fn main() {
    let path = std::env::args().nth(1).expect("usage: gpu_fma FILE.wgsl SOURCE.f32");
    let source = std::fs::read_to_string(&path).expect("read the shader");
    // The source image: width, height (u32) then RGBA f32 pixels.
    let raw = std::fs::read(std::env::args().nth(2).expect("the source image")).expect("read the source");
    let (w, h) = (u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize, u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize);
    let pixels: &[f32] = bytemuck::cast_slice(&raw[8..]);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..wgpu::InstanceDescriptor::new_without_display_handle() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("no Vulkan adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor { label: Some("probe"), required_features: wgpu::Features::empty(), required_limits: wgpu::Limits::default(), ..Default::default() },
    ))
    .expect("device");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let entries: Vec<_> = (0..2)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: binding == 0 }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("probe"), entries: &entries });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("probe"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
    });
    let src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("src"),
        contents: bytemuck::cast_slice(pixels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: 256,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: 256,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &read, 0, 256);
    queue.submit(Some(encoder.finish()));
    let slice = read.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().expect("mapped buffer range");
    let values: &[f32] = bytemuck::cast_slice(&mapped);
    for (i, v) in values.iter().take(40).enumerate() {
        println!("{i} {} {:#010x}", v, v.to_bits());
    }
    let _ = (w, h);
}
