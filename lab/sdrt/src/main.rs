//! sdrt: the command-line tool of open-sbsar (research and batch rendering).
//!
//! Add `--backend vulkan` for the GPU path.
//! `sdrt render FILE OUTDIR` writes every output as a 16-bit PNG named after its identifier;
//! `sdrt node FILE INDEX OUT.png` writes one node's image. FILE is a .sbsar or an .sbsasm.

use open_sbsar::{envc, fxmap, image, ops, package, program, vm};

use std::path::Path;

use program::Program;

fn load(path: &Path) -> Result<(Vec<u8>, String), String> {
    package::load_path(path)
}

fn output_names(xml: &str) -> Vec<(u32, String)> {
    package::output_names(xml)
}

/// Apply `--set ID=v[,v…]` options to the runtime's inputs (ints for int inputs, floats else).
fn apply_sets(rt: &mut ops::Runtime, xml: &str, sets: &[String]) -> Result<(), String> {
    let uids = package::input_uids(xml);
    for (uid, d) in package::string_defaults(xml) {
        rt.strings.insert(uid, d);
    }
    for s in sets {
        let (id, val) = s.split_once('=').ok_or("--set ID=value")?;
        // files without the input (a fixed size, no $outputsize) keep their own value, as sbsrender does
        let Some(uid) = uids.iter().find(|(i, _)| i == id).map(|(_, u)| *u) else {
            eprintln!("warning: no input {id}, ignored");
            continue;
        };
        if rt.strings.contains_key(&uid) {
            rt.strings.insert(uid, val.to_string());
            continue;
        }
        let parts: Vec<&str> = val.split(',').collect();
        let v = match rt.inputs.get(&uid) {
            Some(vm::Value::I(..)) | None => vm::Value::int(&parts.iter().map(|p| p.parse().unwrap_or(0)).collect::<Vec<i32>>()),
            _ => vm::Value::float(&parts.iter().map(|p| p.parse().unwrap_or(0.0)).collect::<Vec<f32>>()),
        };
        rt.inputs.insert(uid, v);
    }
    Ok(())
}

fn save(img: &image::Image, path: &Path) -> Result<(), String> {
    use ::image::{ImageBuffer, Luma, Rgba};
    // Research probes: preserve signed/HDR values and every f32 bit, without PNG quantization.
    // Layout matches the `raw` image decoder command: u32 width/height, then RGBA f32 pixels.
    if path.extension().is_some_and(|ext| ext == "f32") {
        let mut bytes = Vec::with_capacity(8 + img.px.len() * 16);
        bytes.extend_from_slice(&(img.w as u32).to_le_bytes());
        bytes.extend_from_slice(&(img.h as u32).to_le_bytes());
        for pixel in &img.px {
            for value in pixel { bytes.extend_from_slice(&value.to_le_bytes()); }
        }
        return std::fs::write(path, bytes).map_err(|e| e.to_string());
    }
    // sbsrender writes 16-bit PNGs with the same fixed-point conversion as its UNORM16 storage
    // (all 32,640 measured codes; plain rounding is one code higher on ≈3%: scratch/fb/unormmap.py)
    let q = |v: f32| image::unorm(v, 16) as u16;
    if img.color {
        let px: Vec<u16> = img.px.iter().flat_map(|p| p.map(q)).collect();
        ImageBuffer::<Rgba<u16>, _>::from_raw(img.w as u32, img.h as u32, px).unwrap().save(path).map_err(|e| e.to_string())
    } else {
        let px: Vec<u16> = img.px.iter().map(|p| q(p[0])).collect();
        ImageBuffer::<Luma<u16>, _>::from_raw(img.w as u32, img.h as u32, px).unwrap().save(path).map_err(|e| e.to_string())
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut sets = vec![];
    while let Some(k) = args.iter().position(|a| a == "--set") {
        sets.push(args.remove(k + 1));
        args.remove(k);
    }
    let backend = if let Some(k) = args.iter().position(|a| a == "--backend") {
        args.remove(k);
        if k >= args.len() { eprintln!("--backend requires cpu or vulkan"); std::process::exit(2); }
        args.remove(k)
    } else { "cpu".into() };
    let run = || -> Result<(), String> {
        if backend != "cpu" && backend != "vulkan" { return Err(format!("unknown backend {backend}")); }
        match args.first().map(String::as_str) {
            Some("render") if args.len() == 3 => {
                let (b, xml) = load(Path::new(&args[1]))?;
                let p = Program::parse(b)?;
                let names = output_names(&xml);
                let mut rt = ops::Runtime::new(&p);
                if backend == "vulkan" { rt.enable_gpu()?; }
                apply_sets(&mut rt, &xml, &sets)?;
                rt.run(false);
                std::fs::create_dir_all(&args[2]).map_err(|e| e.to_string())?;
                for o in &p.outputs {
                    let name = names.iter().find(|(u, _)| *u == o.uid).map_or(format!("{}", o.uid), |(_, n)| n.clone());
                    let img = rt.images[o.node].as_ref().ok_or("output not computed")?;
                    save(img, &Path::new(&args[2]).join(format!("{name}.png")))?;
                }
                report(&rt);
                Ok(())
            }
            Some("node") if args.len() == 4 => {
                let (b, xml) = load(Path::new(&args[1]))?;
                let p = Program::parse(b)?;
                let k: usize = args[2].parse().map_err(|_| "bad node index")?;
                let mut rt = ops::Runtime::new(&p);
                if backend == "vulkan" { rt.enable_gpu()?; }
                apply_sets(&mut rt, &xml, &sets)?;
                rt.run_for(&[k]);
                save(rt.images[k].as_ref().ok_or("not computed")?, Path::new(&args[3]))?;
                report(&rt);
                Ok(())
            }
            Some("nodes") if args.len() >= 4 => {
                // several nodes from one run: OUTDIR/n<K>.png
                let (b, xml) = load(Path::new(&args[1]))?;
                let p = Program::parse(b)?;
                let ks: Vec<usize> = args[3..].iter().filter_map(|a| a.parse().ok()).collect();
                let mut rt = ops::Runtime::new(&p);
                if backend == "vulkan" { rt.enable_gpu()?; }
                apply_sets(&mut rt, &xml, &sets)?;
                std::fs::create_dir_all(&args[2]).map_err(|e| e.to_string())?;
                // each node is written as soon as it is computed (memory stays bounded; a run cut
                // short by a timeout still leaves the nodes done so far). SDRT_NODES_F32=1 writes
                // raw n<K>.f32 (every bit, no PNG quantization) instead of 16-bit PNGs.
                let dir = Path::new(&args[2]).to_path_buf();
                let ext = if envc::var("SDRT_NODES_F32").is_ok() { "f32" } else { "png" };
                rt.sink = Some(Box::new(move |k, img| {
                    if let Err(e) = save(img, &dir.join(format!("n{k}.{ext}"))) { eprintln!("node {k}: {e}"); }
                }));
                rt.run_for(&ks);
                report(&rt);
                Ok(())
            }
            Some("stats") if args.len() == 2 => {
                // every computed node: op, size, min / max / mean of channel 0 (and alpha)
                let (b, xml) = load(Path::new(&args[1]))?;
                let p = Program::parse(b)?;
                let mut rt = ops::Runtime::new(&p);
                if backend == "vulkan" { rt.enable_gpu()?; }
                apply_sets(&mut rt, &xml, &sets)?;
                rt.run(true);
                for (k, im) in rt.images.iter().enumerate() {
                    let Some(im) = im else { continue };
                    let (mut lo, mut hi, mut sum, mut asum) = (f32::MAX, f32::MIN, 0f64, 0f64);
                    for q in &im.px {
                        lo = lo.min(q[0]);
                        hi = hi.max(q[0]);
                        sum += q[0] as f64;
                        asum += q[3] as f64;
                    }
                    let n = im.px.len() as f64;
                    println!("{} op {:#04x} {}x{} min {} max {} mean {:.5} alpha {:.5}", k, p.nodes[k].op, im.w, im.h, lo, hi, sum / n, asum / n);
                }
                report(&rt);
                Ok(())
            }
            Some("cmp") if args.len() == 3 => {
                // Per pixel, take the largest absolute channel difference; report its max/mean.
                let a = ::image::open(&args[1]).map_err(|e| e.to_string())?;
                let b = ::image::open(&args[2]).map_err(|e| e.to_string())?;
                let n = (a.color().channel_count()).min(b.color().channel_count()) as usize;
                let (a, b) = (a.to_rgba32f(), b.to_rgba32f());
                if a.dimensions() != b.dimensions() {
                    println!("size {:?} {:?}", a.dimensions(), b.dimensions());
                    return Ok(());
                }
                let ch = if n >= 3 { n } else { 1 };
                let (mut worst, mut total, mut at) = (0f32, 0f64, (0, 0));
                for (x, y, p) in a.enumerate_pixels() {
                    let q = b.get_pixel(x, y);
                    let d = (0..ch).map(|c| (p.0[c] - q.0[c]).abs()).fold(0f32, f32::max);
                    total += d as f64;
                    if d > worst {
                        worst = d;
                        at = (x, y);
                    }
                }
                let (w, h) = a.dimensions();
                println!("{} {} {} {}", worst, total / (w as f64 * h as f64), at.0, at.1);
                Ok(())
            }
            Some("raw") if args.len() == 3 => {
                // any image as width, height (u32) then RGBA f32 pixels, little endian
                let a = ::image::open(&args[1]).map_err(|e| e.to_string())?.to_rgba32f();
                let mut out = Vec::with_capacity(8 + a.as_raw().len() * 4);
                out.extend_from_slice(&a.width().to_le_bytes());
                out.extend_from_slice(&a.height().to_le_bytes());
                for v in a.as_raw() {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(&args[2], out).map_err(|e| e.to_string())
            }
            Some("inputs") if args.len() == 2 => {
                // image inputs of every node (for tools/sdbisect.py)
                let (b, _) = load(Path::new(&args[1]))?;
                let p = Program::parse(b)?;
                for n in &p.nodes {
                    let mut ins = n.inputs.clone();
                    for (_, s) in &n.params {
                        if let program::Slot::Input(k) = s {
                            ins.push(*k);
                        }
                    }
                    println!("{}: {}", n.index, ins.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(" "));
                }
                Ok(())
            }
            Some("curves") if args.len() == 2 => {
                // color curve nodes: point counts read with each curve's own length and with the
                // stride of the longest
                let (b, _) = load(Path::new(&args[1]))?;
                let p = Program::parse(b)?;
                for n in p.nodes.iter().filter(|n| n.op == 0x2C && n.color) {
                    let mut own = vec![];
                    let mut o = n.data;
                    for _ in 0..5 {
                        let k = p.w(o);
                        own.push(k);
                        o += 4 + 24 * k.min(256) as usize;
                    }
                    let stride: Vec<u32> = (0..5).map(|i| p.w(n.data + i * (4 + 24 * n.count as usize))).collect();
                    println!("{} count {} own {:?} stride {:?}", n.index, n.count, own, stride);
                }
                Ok(())
            }
            Some("sizes") if args.len() == 2 => {
                // nodes with a size function: index, block offset of the slot word, function offset
                // (for tools/sdmulti.small_copy)
                let (b, _) = load(Path::new(&args[1]))?;
                let p = Program::parse(b)?;
                for n in &p.nodes {
                    if let program::Slot::Fn(f) = n.size {
                        println!("{} {} {}", n.index, n.size_at, f);
                    }
                }
                Ok(())
            }
            Some("fn") if args.len() == 3 => {
                // a decoded function: one line per instruction (index, op, type, operands)
                let (b, _) = load(Path::new(&args[1]))?;
                let p = Program::parse(b)?;
                let off: usize = args[2].parse().map_err(|_| "bad offset")?;
                let f = vm::Func::decode(&p, off);
                for (k, i) in f.ins.iter().enumerate() {
                    println!("{k} {} {} {:?}", i.op, i.ty, i.ops);
                }
                Ok(())
            }
            Some("output-nodes") if args.len() == 2 => {
                let (b, _) = load(Path::new(&args[1]))?;
                let p = Program::parse(b)?;
                for o in &p.outputs {
                    if o.flags & 0x10000 != 0 { println!("{}", o.node); }
                }
                Ok(())
            }
            Some("ops") if args.len() == 2 => {
                // one line per node: index, op, flags, color (for tools/sdvers.py)
                let (b, _) = load(Path::new(&args[1]))?;
                let p = Program::parse(b)?;
                for n in &p.nodes {
                    println!("{} {} {} {}", n.index, n.op, n.flags, n.color as u8);
                }
                Ok(())
            }
            Some("info") if args.len() == 3 => {
                let (b, _) = load(Path::new(&args[1]))?;
                let p = Program::parse(b)?;
                let k: usize = args[2].parse().map_err(|_| "bad node index")?;
                println!("{:#?}", p.nodes[k]);
                Ok(())
            }
            _ => Err("usage: sdrt render FILE OUTDIR | node FILE INDEX OUT.png | inputs FILE | info FILE INDEX".into()),
        }
    };
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn report(rt: &ops::Runtime) {
    if crate::envc::var("SDRT_FXPROF").as_deref() == Ok("2") {
        fxmap::prof_fn_report();
    }
    if std::env::var("SDRT_TIME").is_ok() {
        for (op, t) in &rt.op_time {
            eprintln!("time: op {op:#04x} {t:.2}s");
        }
    }
    if !rt.gpu_ops.is_empty() && std::env::var("SDRT_GPU_LOG").is_ok() {
        let row: Vec<String> = rt.gpu_ops.iter().map(|(op, (g, c))| format!("{op:#04x} {g}/{}", g + c)).collect();
        eprintln!("GPU coverage (op gpu/all): {}", row.join(", "));
    }
    let mut kinds: std::collections::BTreeMap<&str, usize> = Default::default();
    for (_, why) in &rt.unsupported {
        *kinds.entry(why.as_str()).or_default() += 1;
    }
    for (why, n) in kinds {
        eprintln!("unsupported: {n} × {why}");
    }
}
