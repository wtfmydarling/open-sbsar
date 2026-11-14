//! Render every image output of a .sbsar to 16-bit PNGs with the open-sbsar API.
//!
//! cargo run --release --example render -- MATERIAL.sbsar OUTDIR [LOG2SIZE] [cpu]
use open_sbsar::{Package, Parameters, Renderer, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (path, outdir) = (&args[0], std::path::Path::new(&args[1]));
    let size: i32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let package = Package::open(path)?;
    let graph = &package.graphs()[0];
    println!("{} graph(s); \"{}\", format version {}", package.graphs().len(), graph.label().unwrap_or("?"), graph.format_version());
    for input in graph.inputs() {
        println!("  input  {:<28} {:?} default {:?} range {:?}..{:?}", input.identifier, input.kind, input.default, input.min, input.max);
    }
    let mut params = Parameters::new();
    params.set("$outputsize", Value::Int(vec![size, size]));
    let mut renderer = if args.get(3).is_some_and(|a| a == "cpu") { Renderer::new_cpu() } else { Renderer::new_gpu().unwrap_or_else(|_| Renderer::new_cpu()) };
    // the same renderer renders twice: the second render reuses the GPU context and its shaders
    for pass in 0..2 {
        let t = std::time::Instant::now();
        let outputs = renderer.render(graph, &params, None)?;
        println!("render {pass}: {} outputs in {:.2} s on the {}", outputs.len(), t.elapsed().as_secs_f64(), if renderer.is_gpu() { "GPU" } else { "CPU" });
        if pass == 1 {
            std::fs::create_dir_all(outdir)?;
            for out in &outputs {
                println!("  output {:<28} {}x{} {} ch {:?}", out.identifier, out.width, out.height, out.channels, out.format);
                let (w, h) = (out.width as u32, out.height as u32);
                if out.channels == 4 {
                    image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(w, h, out.to_rgba16()).unwrap().save(outdir.join(format!("{}.png", out.identifier)))?;
                } else {
                    let gray: Vec<u16> = out.to_rgba16().chunks(4).map(|p| p[0]).collect();
                    image::ImageBuffer::<image::Luma<u16>, _>::from_raw(w, h, gray).unwrap().save(outdir.join(format!("{}.png", out.identifier)))?;
                }
            }
        }
    }
    for w in &renderer.warnings {
        println!("warning: {w}");
    }
    Ok(())
}
