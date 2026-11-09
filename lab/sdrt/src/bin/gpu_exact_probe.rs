//! A research probe for the prelude's correctly rounded helpers: it runs a WGSL snippet over
//! many values on the GPU and compares the result with the CPU's own (correctly rounded)
//! f32::sqrt and f32::div. The values are generated here (full-mantissa random, edge cases and
//! every exponent), so the comparison covers a wide range in one run.
//!
//! Usage: gpu_exact_probe <wgsl file>   (the snippet must write one vec4 per input pixel)

use wgpu::util::DeviceExt;

fn main() {
    let path = std::env::args().nth(1).expect("usage: gpu_exact_probe FILE.wgsl");
    let source = std::fs::read_to_string(&path).expect("read the shader");
    // The probe values: full-mantissa random floats in many ranges, plus edge cases.
    let mut values: Vec<f32> = Vec::new();
    let mut state: u32 = 0x12345678;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for _ in 0..60000 {
        let bits = next();
        let exp = (bits >> 20) & 0xfe; // an even exponent
        values.push(f32::from_bits((exp << 23) | (bits & 0x007fffff)));
    }
    for e in 0..255u32 {
        for m in [0u32, 1, 0x400000, 0x7fffff, 0x123456] {
            values.push(f32::from_bits((e << 23) | m));
        }
    }
    // Every mantissa pattern of a few exponents, and the values around the powers of two.
    for e in [1u32, 2, 100, 126, 127, 128, 200, 253, 254] {
        for m in 0..1024u32 {
            values.push(f32::from_bits((e << 23) | (m * 8191)));
        }
    }
    // The neighbours of the subnormal boundary and of zero.
    for m in [0u32, 1, 2, 3, 0x7ffffe, 0x7fffff] {
        values.push(f32::from_bits(m));
        values.push(f32::from_bits(m | 0x80000000));
        values.push(f32::from_bits(0x00800000u32 | m));
    }
    for v in [0.0f32, 1.0, 2.0, 3.0, 4.0, 0.5, 1e-30, 1e30, 3.0, 1.0 / 3.0, 65535.0] {
        values.push(v);
    }
    let count = values.len();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("no Vulkan adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor { label: Some("probe"), required_features: wgpu::Features::empty(), required_limits: wgpu::Limits::default() },
        None,
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
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("probe"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: "main",
    });
    // The source image: one value per pixel, in the four channels.
    let mut pixels: Vec<f32> = Vec::with_capacity(count * 4);
    for &v in &values {
        pixels.extend_from_slice(&[v, v, v, v]);
    }
    let src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("src"),
        contents: bytemuck::cast_slice(&pixels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = (count * 16) as u64;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: bytes,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: bytes,
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
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &read, 0, bytes);
    queue.submit(Some(encoder.finish()));
    let slice = read.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range();
    let got: &[f32] = bytemuck::cast_slice(&mapped);
    // The comparison: the GPU's helper against the CPU's own correctly rounded operation.
    let mut sqrt_bad = 0usize;
    let mut div_bad = 0usize;
    let mut gpu_sqrt_bad = 0usize;
    let mut gpu_div_bad = 0usize;
    for i in 0..count {
        let x = values[i];
        let g = [got[i * 4], got[i * 4 + 1], got[i * 4 + 2], got[i * 4 + 3]];
        let c_sqrt = x.sqrt();
        let c_div = x / 3.0f32;
        if g[0].to_bits() != c_sqrt.to_bits() && x > 0.0 {
            if sqrt_bad < 8 { eprintln!("sqrt mismatch x={:e} (#{:#010x}) gpu={:e} (#{:#010x}) cpu={:e} (#{:#010x})", x, x.to_bits(), g[0], g[0].to_bits(), c_sqrt, c_sqrt.to_bits()); }
            sqrt_bad += 1;
        }
        if g[1].to_bits() != c_div.to_bits() {
            if div_bad < 8 { eprintln!("div mismatch x={:e} (#{:#010x}) gpu={:e} (#{:#010x}) cpu={:e} (#{:#010x})", x, x.to_bits(), g[1], g[1].to_bits(), c_div, c_div.to_bits()); }
            div_bad += 1;
        }
        // The driver's own sqrt and division, for comparison.
        if g[2].to_bits() != c_sqrt.to_bits() && x > 0.0 { gpu_sqrt_bad += 1; }
        if g[3].to_bits() != c_div.to_bits() { gpu_div_bad += 1; }
    }
    eprintln!("values: {count}");
    eprintln!("sqrt_rn mismatches: {sqrt_bad}   (the driver's own sqrt: {gpu_sqrt_bad})");
    eprintln!("div_rn  mismatches: {div_bad}   (the driver's own division: {gpu_div_bad})");
    if sqrt_bad > 0 || div_bad > 0 {
        std::process::exit(1);
    }
}
