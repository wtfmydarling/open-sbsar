//! Stress-tests incremental rendering (`Instance`) on CPU and GPU at $outputsize=7,7 across
//! 6 scenarios:
//! 1. Output subsets
//! 2. String and image parameter changes
//! 3. Two independent instances (same graph & different graphs) interleaved on one renderer
//! 4. Cache eviction during execution (1, 2, 8 MB budgets)
//! 5. Failed renders (non-existing output) followed by normal render
//! 6. Setting a parameter to the same value again (must reuse everything)
//!
//! cargo run --release --example incr_scenarios -- [cpu|gpu|both] [FILE.sbsar...]
use open_sbsar::{
    Graph, InputImage, InputKind, Instance, OutputImage, Package, Parameters, PixelFormat,
    Renderer, Result, Value,
};

const DEFAULT_MATERIALS: &[&str] = &[
    "/home/andrew/Documents/SBSAR/Materials/2555_wool.sbsar",
    "/home/andrew/Documents/SBSAR/Materials/3cm-bricks.sbsar",
    "/home/andrew/Documents/SBSAR/Materials/abstract_01.sbsar",
    "/home/andrew/Documents/SBSAR/Materials/acousticfoamsubstance001.sbsar",
    "/home/andrew/Documents/SBSAR/Materials/balloon.sbsar",
    "/home/andrew/Documents/SBSAR/Materials/bark_old_ginko.sbsar",
    "/home/andrew/Documents/SBSAR/Materials/bark.sbsar",
    "/home/andrew/Documents/SBSAR/TextureGenerators/manhole_sbsar_1.sbsar",
];

const DEFAULT_STRING_FILE: &str =
    "/home/andrew/Documents/SBSAR/TextureGenerators/manhole_sbsar_1.sbsar";
const DEFAULT_IMAGE_FILES: &[&str] = &[
    "/home/andrew/Documents/SBSAR/Filters/dust_splatter.sbsar",
    "/home/andrew/Documents/SBSAR/Filters/ask_pathfill.sbsar",
];

#[derive(Default)]
struct Counts {
    matched: usize,
    mismatch: usize,
    both_error: usize,
    inc_error: usize,
    full_error: usize,
    skipped: usize,
}

impl Counts {
    fn failed(&self) -> bool {
        self.mismatch + self.both_error + self.inc_error + self.full_error > 0
    }
    fn print_summary(&self, label: &str) {
        println!(
            "=== {label} SUMMARY: MATCH={} MISMATCH={} BOTH-ERROR={} INC-ERROR={} FULL-ERROR={} SKIPPED={} ===",
            self.matched, self.mismatch, self.both_error, self.inc_error, self.full_error, self.skipped
        );
    }
    fn add(&mut self, other: &Counts) {
        self.matched += other.matched;
        self.mismatch += other.mismatch;
        self.both_error += other.both_error;
        self.inc_error += other.inc_error;
        self.full_error += other.full_error;
        self.skipped += other.skipped;
    }
}

fn renderer(gpu: bool) -> Renderer {
    if gpu {
        Renderer::new_gpu().expect("no GPU")
    } else {
        Renderer::new_cpu()
    }
}

fn compare(
    counts: &mut Counts,
    label: &str,
    inc: Result<Vec<OutputImage>>,
    full: Result<Vec<OutputImage>>,
) {
    match (&inc, &full) {
        (Ok(a), Ok(b)) => {
            let eq = a.len() == b.len()
                && a.iter().zip(b).all(|(x, y)| {
                    x.identifier == y.identifier
                        && x.width == y.width
                        && x.height == y.height
                        && x.format == y.format
                        && x.data == y.data
                });
            if eq {
                counts.matched += 1;
                println!("  MATCH      {label}");
            } else {
                counts.mismatch += 1;
                println!("  MISMATCH   {label} (len {} vs {})", a.len(), b.len());
                for (x, y) in a.iter().zip(b.iter()) {
                    if x.identifier != y.identifier
                        || x.width != y.width
                        || x.height != y.height
                        || x.format != y.format
                        || x.data != y.data
                    {
                        println!(
                            "    diff: {} ({}x{} {:?}, {} bytes) vs {} ({}x{} {:?}, {} bytes)",
                            x.identifier,
                            x.width,
                            x.height,
                            x.format,
                            x.data.len(),
                            y.identifier,
                            y.width,
                            y.height,
                            y.format,
                            y.data.len()
                        );
                    }
                }
            }
        }
        (Err(ei), Err(ef)) => {
            counts.both_error += 1;
            println!("  BOTH-ERROR {label}: inc={ei} | full={ef}");
        }
        (Err(ei), Ok(_)) => {
            counts.inc_error += 1;
            println!("  INC-ERROR  {label}: {ei}");
        }
        (Ok(_), Err(ef)) => {
            counts.full_error += 1;
            println!("  FULL-ERROR {label}: {ef}");
        }
    }
}

fn first_param_change(graph: &Graph) -> Option<(String, Value)> {
    for i in graph.inputs() {
        if i.identifier.starts_with('$') {
            continue;
        }
        let def: Vec<f64> = i
            .default
            .as_deref()
            .unwrap_or("")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let pick = |k: usize, t: f64| -> f64 {
            let lo = i.min.as_ref().and_then(|m| m.get(k).or(m.first()).copied()).unwrap_or(0.0);
            let hi = i.max.as_ref().and_then(|m| m.get(k).or(m.first()).copied()).unwrap_or(1.0);
            lo + t * (hi - lo)
        };
        match i.kind {
            InputKind::Float(n) if n >= 1 => {
                let mut v: Vec<f32> = (0..n).map(|k| pick(k, 0.3) as f32).collect();
                if v.iter().zip(&def).all(|(a, b)| (*a as f64 - b).abs() < 1e-6) {
                    v = (0..n).map(|k| pick(k, 0.7) as f32).collect();
                }
                return Some((i.identifier.clone(), Value::Float(v)));
            }
            InputKind::Int(n) if n >= 1 => {
                let d = def.first().copied().unwrap_or(0.0) as i32;
                let x = if let Some((o, _)) = i.options.iter().find(|(o, _)| *o != d) {
                    *o
                } else {
                    let m = pick(0, 0.5).round() as i32;
                    if m != d { m } else { d + 1 }
                };
                let mut v: Vec<i32> = (0..n).map(|k| def.get(k).copied().unwrap_or(0.0) as i32).collect();
                v[0] = x;
                return Some((i.identifier.clone(), Value::Int(v)));
            }
            _ => continue,
        }
    }
    None
}

fn param_changes(graph: &Graph) -> Vec<(String, Value)> {
    let mut out = vec![];
    for i in graph.inputs() {
        if out.len() >= 3 || i.identifier.starts_with('$') {
            continue;
        }
        let def: Vec<f64> = i
            .default
            .as_deref()
            .unwrap_or("")
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let pick = |k: usize, t: f64| -> f64 {
            let lo = i.min.as_ref().and_then(|m| m.get(k).or(m.first()).copied()).unwrap_or(0.0);
            let hi = i.max.as_ref().and_then(|m| m.get(k).or(m.first()).copied()).unwrap_or(1.0);
            lo + t * (hi - lo)
        };
        match i.kind {
            InputKind::Float(n) if n >= 1 => {
                let mut v: Vec<f32> = (0..n).map(|k| pick(k, 0.25) as f32).collect();
                if v.iter().zip(&def).all(|(a, b)| (*a as f64 - b).abs() < 1e-6) {
                    v = (0..n).map(|k| pick(k, 0.75) as f32).collect();
                }
                out.push((i.identifier.clone(), Value::Float(v)));
            }
            InputKind::Int(n) if n >= 1 => {
                let d = def.first().copied().unwrap_or(0.0) as i32;
                let x = if let Some((o, _)) = i.options.iter().find(|(o, _)| *o != d) {
                    *o
                } else {
                    let m = pick(0, 0.5).round() as i32;
                    if m != d { m } else { d + 1 }
                };
                let mut v: Vec<i32> = (0..n).map(|k| def.get(k).copied().unwrap_or(0.0) as i32).collect();
                v[0] = x;
                out.push((i.identifier.clone(), Value::Int(v)));
            }
            _ => continue,
        }
    }
    out.push(("$randomseed".into(), Value::Int(vec![17])));
    out
}

fn base_params() -> Parameters {
    let mut p = Parameters::new();
    p.set("$outputsize", Value::Int(vec![7, 7]));
    p
}

fn make_gradient(w: usize, h: usize, ch: usize, phase: f32) -> InputImage {
    let mut data = Vec::with_capacity(w * h * ch);
    for y in 0..h {
        for x in 0..w {
            let u = x as f32 / w.max(1) as f32;
            let v = y as f32 / h.max(1) as f32;
            if ch == 1 {
                data.push(((u + v + phase) * 0.5).fract());
            } else {
                data.push((u + phase).fract());
                data.push((v + phase * 0.5).fract());
                data.push(((u * 0.5 + v * 0.5) + phase).fract());
                data.push(1.0);
            }
        }
    }
    InputImage { width: w, height: h, channels: ch, data, storage: PixelFormat::U8 }
}

fn make_checker(w: usize, h: usize, ch: usize) -> InputImage {
    let mut data = Vec::with_capacity(w * h * ch);
    for y in 0..h {
        for x in 0..w {
            let c = if ((x / 8) + (y / 8)) % 2 == 0 { 0.85 } else { 0.15 };
            if ch == 1 {
                data.push(c);
            } else {
                data.extend_from_slice(&[c, 1.0 - c, 0.5 * c, 1.0]);
            }
        }
    }
    InputImage { width: w, height: h, channels: ch, data, storage: PixelFormat::U8 }
}

fn run_backend(gpu: bool, paths: &[String]) -> Counts {
    let bname = if gpu { "GPU" } else { "CPU" };
    println!("================ {bname} ================");
    let mut counts = Counts::default();

    let mut loaded: Vec<(String, Package)> = Vec::new();
    for p in paths {
        let name = std::path::Path::new(p).file_name().unwrap().to_string_lossy().to_string();
        match Package::open(p) {
            Ok(pkg) => loaded.push((name, pkg)),
            Err(e) => {
                println!("SKIPPED {name}: cannot open: {e}");
                counts.skipped += 1;
            }
        }
    }

    // 1. Output subsets
    println!("[1] Output subsets ({bname})");
    for (name, pkg) in &loaded {
        let g = &pkg.graphs()[0];
        let outs: Vec<&str> = g.outputs().iter().filter(|o| o.is_image).map(|o| o.identifier.as_str()).collect();
        if outs.len() < 2 {
            println!("  SKIPPED {name}: fewer than 2 image outputs");
            counts.skipped += 1;
            continue;
        }
        let a = outs[0];
        let b = outs[1];
        let (mut inc_r, mut full_r) = (renderer(gpu), renderer(gpu));
        let mut inst = Instance::new(g);
        let mut params = base_params();
        let ab = [a, b];
        let inc_ab = inst.render(&mut inc_r, g, &params, Some(&ab));
        let full_ab = full_r.render(g, &params, Some(&ab));
        compare(&mut counts, &format!("{name} init [A,B]"), inc_ab, full_ab);

        // Change $randomseed (affects both in most materials) and/or first param while rendering only A
        params.set("$randomseed", Value::Int(vec![42]));
        if let Some((pid, pval)) = first_param_change(g) {
            params.set(&pid, pval);
        }
        let inc_a = inst.render(&mut inc_r, g, &params, Some(&[a]));
        let full_a = full_r.render(g, &params, Some(&[a]));
        compare(&mut counts, &format!("{name} param+seed -> only [A]"), inc_a, full_a);

        // Render only B with nothing else changed!
        let inc_b = inst.render(&mut inc_r, g, &params, Some(&[b]));
        let full_b = full_r.render(g, &params, Some(&[b]));
        compare(&mut counts, &format!("{name} unchanged -> only [B]"), inc_b, full_b);

        // Render all
        let inc_all = inst.render(&mut inc_r, g, &params, None);
        let full_all = full_r.render(g, &params, None);
        compare(&mut counts, &format!("{name} unchanged -> all"), inc_all, full_all);
    }

    // 2. String parameter changes and image parameter changes
    println!("[2] String & Image parameter changes ({bname})");
    if let Ok(pkg) = Package::open(DEFAULT_STRING_FILE) {
        let g = &pkg.graphs()[0];
        if let Some(str_in) = g.inputs().iter().find(|i| matches!(i.kind, InputKind::String)) {
            let (mut inc_r, mut full_r) = (renderer(gpu), renderer(gpu));
            let mut inst = Instance::new(g);
            let mut params = base_params();
            for s in ["HELLO", "WORLD", "WORLD", ""] {
                params.set(&str_in.identifier, Value::String(s.to_string()));
                let inc = inst.render(&mut inc_r, g, &params, None);
                let full = full_r.render(g, &params, None);
                compare(&mut counts, &format!("manhole_sbsar_1 str={s:?}"), inc, full);
            }
        } else {
            println!("  SKIPPED string test: no string input");
            counts.skipped += 1;
        }
    } else {
        println!("  SKIPPED string test: cannot open {DEFAULT_STRING_FILE}");
        counts.skipped += 1;
    }

    for img_path in DEFAULT_IMAGE_FILES {
        let name = std::path::Path::new(img_path).file_name().unwrap().to_string_lossy().to_string();
        if let Ok(pkg) = Package::open(img_path) {
            let g = &pkg.graphs()[0];
            let img_inputs: Vec<String> = g
                .inputs()
                .iter()
                .filter(|i| matches!(i.kind, InputKind::Image))
                .map(|i| i.identifier.clone())
                .collect();
            if img_inputs.is_empty() {
                println!("  SKIPPED image test {name}: no image input");
                counts.skipped += 1;
                continue;
            }
            let (mut inc_r, mut full_r) = (renderer(gpu), renderer(gpu));
            let mut inst = Instance::new(g);
            let mut params = base_params();
            for id in &img_inputs {
                params.set_image(id, make_gradient(64, 64, 4, 0.0));
            }
            compare(
                &mut counts,
                &format!("{name} image=grad1"),
                inst.render(&mut inc_r, g, &params, None),
                full_r.render(g, &params, None),
            );
            for id in &img_inputs {
                params.set_image(id, make_checker(64, 64, 4));
            }
            compare(
                &mut counts,
                &format!("{name} image=checker"),
                inst.render(&mut inc_r, g, &params, None),
                full_r.render(g, &params, None),
            );
            // Same image again (newly constructed InputImage with identical pixels)
            for id in &img_inputs {
                params.set_image(id, make_checker(64, 64, 4));
            }
            compare(
                &mut counts,
                &format!("{name} image=checker_again"),
                inst.render(&mut inc_r, g, &params, None),
                full_r.render(g, &params, None),
            );
        } else {
            println!("  SKIPPED image test {name}: cannot open");
            counts.skipped += 1;
        }
    }

    // 3. Two independent instances (same graph & different graphs) sharing one renderer, interleaved
    println!("[3] Interleaved instances on one renderer ({bname})");
    if loaded.len() >= 2 {
        let (n1, p1) = &loaded[0];
        let (n2, p2) = &loaded[1];
        let g1 = &p1.graphs()[0];
        let g2 = &p2.graphs()[0];
        let mut shared_r = renderer(gpu);
        let mut full_r = renderer(gpu);

        // Same graph, two instances
        let mut i1a = Instance::new(g1);
        let mut i1b = Instance::new(g1);
        let mut pa = base_params();
        let mut pb = base_params();
        pb.set("$randomseed", Value::Int(vec![99]));
        compare(&mut counts, &format!("{n1} same-graph i1a#1"), i1a.render(&mut shared_r, g1, &pa, None), full_r.render(g1, &pa, None));
        compare(&mut counts, &format!("{n1} same-graph i1b#1"), i1b.render(&mut shared_r, g1, &pb, None), full_r.render(g1, &pb, None));
        pa.set("$randomseed", Value::Int(vec![7]));
        compare(&mut counts, &format!("{n1} same-graph i1a#2"), i1a.render(&mut shared_r, g1, &pa, None), full_r.render(g1, &pa, None));
        compare(&mut counts, &format!("{n1} same-graph i1b#2"), i1b.render(&mut shared_r, g1, &pb, None), full_r.render(g1, &pb, None));

        // Different graphs, two instances
        let mut j1 = Instance::new(g1);
        let mut j2 = Instance::new(g2);
        let mut q1 = base_params();
        let mut q2 = base_params();
        compare(&mut counts, &format!("{n1}/{n2} diff-graph j1#1"), j1.render(&mut shared_r, g1, &q1, None), full_r.render(g1, &q1, None));
        compare(&mut counts, &format!("{n1}/{n2} diff-graph j2#1"), j2.render(&mut shared_r, g2, &q2, None), full_r.render(g2, &q2, None));
        q1.set("$randomseed", Value::Int(vec![11]));
        q2.set("$randomseed", Value::Int(vec![22]));
        compare(&mut counts, &format!("{n1}/{n2} diff-graph j1#2"), j1.render(&mut shared_r, g1, &q1, None), full_r.render(g1, &q1, None));
        compare(&mut counts, &format!("{n1}/{n2} diff-graph j2#2"), j2.render(&mut shared_r, g2, &q2, None), full_r.render(g2, &q2, None));
    }

    // 4. Eviction during execution (1, 2, 8 MB)
    println!("[4] Cache budget eviction ({bname})");
    for (name, pkg) in loaded.iter().take(2) {
        let g = &pkg.graphs()[0];
        let (mut inc_r, mut full_r) = (renderer(gpu), renderer(gpu));
        for mb in [1usize, 2, 8] {
            let mut inst = Instance::new(g);
            inst.set_cache_budget(mb << 20);
            let mut params = base_params();
            compare(
                &mut counts,
                &format!("{name} budget={mb}MB init"),
                inst.render(&mut inc_r, g, &params, None),
                full_r.render(g, &params, None),
            );
            for (pid, pval) in param_changes(g).into_iter().take(2) {
                params.set(&pid, pval);
                compare(
                    &mut counts,
                    &format!("{name} budget={mb}MB {pid}"),
                    inst.render(&mut inc_r, g, &params, None),
                    full_r.render(g, &params, None),
                );
            }
        }
    }

    // 5. Failed renders: non-existing output, then normal render
    println!("[5] Failed render recovery ({bname})");
    for (name, pkg) in &loaded {
        let g = &pkg.graphs()[0];
        let (mut inc_r, mut full_r) = (renderer(gpu), renderer(gpu));
        let mut inst = Instance::new(g);
        let params = base_params();
        let _ = inst.render(&mut inc_r, g, &params, None);
        let bad = inst.render(&mut inc_r, g, &params, Some(&["__nonexistent_output__"]));
        if bad.is_ok() {
            counts.mismatch += 1;
            println!("  MISMATCH   {name} nonexistent output unexpectedly succeeded");
        }
        compare(
            &mut counts,
            &format!("{name} after failed render"),
            inst.render(&mut inc_r, g, &params, None),
            full_r.render(g, &params, None),
        );
    }

    // 6. Setting a parameter to the same value again: must reuse everything and still match
    println!("[6] Same parameter value again ({bname})");
    for (name, pkg) in &loaded {
        let g = &pkg.graphs()[0];
        let (mut inc_r, mut full_r) = (renderer(gpu), renderer(gpu));
        let mut inst = Instance::new(g);
        let mut params = base_params();
        let _ = inst.render(&mut inc_r, g, &params, None);
        if let Some((pid, pval)) = first_param_change(g) {
            params.set(&pid, pval.clone());
            let _ = inst.render(&mut inc_r, g, &params, None);
            params.set(&pid, pval);
        } else {
            params.set("$randomseed", Value::Int(vec![0]));
        }
        let inc = inst.render(&mut inc_r, g, &params, None);
        let st = inst.last_stats();
        let full = full_r.render(g, &params, None);
        if st.reused == 0 {
            counts.mismatch += 1;
            println!(
                "  MISMATCH   {name} same-value: expected reuse, got recomputed={} reused={}",
                st.recomputed, st.reused
            );
        } else {
            compare(
                &mut counts,
                &format!("{name} same-value (recomputed={}, reused={})", st.recomputed, st.reused),
                inc,
                full,
            );
        }
    }

    counts.print_summary(bname);
    counts
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mode, files) = match args.first().map(String::as_str) {
        Some("cpu" | "gpu" | "both") => (args[0].as_str(), args[1..].to_vec()),
        _ => ("both", args),
    };
    let files: Vec<String> = if files.is_empty() {
        DEFAULT_MATERIALS.iter().map(|s| s.to_string()).collect()
    } else {
        files
    };

    let mut total = Counts::default();
    if mode == "cpu" || mode == "both" {
        total.add(&run_backend(false, &files));
    }
    if mode == "gpu" || mode == "both" {
        total.add(&run_backend(true, &files));
    }
    total.print_summary("TOTAL");
    std::process::exit(if total.failed() { 1 } else { 0 });
}
