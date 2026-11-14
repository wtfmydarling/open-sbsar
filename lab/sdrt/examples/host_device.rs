//! Render on a wgpu device created by the host application (here: by this example).
//!
//! cargo run --release --example host_device -- MATERIAL.sbsar
use open_sbsar::wgpu;
use open_sbsar::{Package, Parameters, Renderer, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: host_device MATERIAL.sbsar");
    // the host's own device (any backend the host uses; Vulkan here)
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..wgpu::InstanceDescriptor::new_without_display_handle() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut limits = wgpu::Limits::default();
    limits.max_storage_buffer_binding_size = adapter.limits().max_storage_buffer_binding_size;
    limits.max_buffer_size = adapter.limits().max_buffer_size;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() }))?;

    let package = Package::open(&path)?;
    let graph = &package.graphs()[0];
    let mut renderer = Renderer::with_device(device.clone(), queue.clone())?;
    let mut params = Parameters::new();
    params.set("$outputsize", Value::Int(vec![9, 9]));
    for seed in 0..3 {
        params.set("$randomseed", Value::Int(vec![seed]));
        let t = std::time::Instant::now();
        let outputs = renderer.render(graph, &params, None)?;
        let sum: f64 = outputs.iter().map(|o| o.as_f32().iter().map(|&v| v as f64).sum::<f64>()).sum();
        println!("seed {seed}: {} outputs in {:.2} s (checksum {sum:.3})", outputs.len(), t.elapsed().as_secs_f64());
    }
    // the host keeps using its device afterwards
    let _ = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
    Ok(())
}
