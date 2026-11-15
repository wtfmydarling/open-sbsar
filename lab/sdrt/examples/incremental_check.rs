//! Checks incremental rendering: after each single-parameter change, `Instance::render` must
//! return the same bytes as a full `Renderer::render` with the same parameters.
//!
//! cargo run --release --example incremental_check -- [cpu|gpu] LOG2SIZE FILE.sbsar...
//! Exit code 1 on any mismatch.
use open_sbsar::{Graph, InputKind, Instance, Package, Parameters, Renderer, Value};
use std::time::Instant;

fn renderer(gpu: bool) -> Renderer {
    if gpu { Renderer::new_gpu().expect("no GPU") } else { Renderer::new_cpu() }
}

/// Single-parameter changes: up to 5 graph parameters moved inside their range, the random
/// seed, the output size, and one parameter set back to its default.
fn changes(graph: &Graph, size: i32) -> Vec<(String, Value)> {
    let mut out = vec![];
    let mut first: Option<(String, Value)> = None;
    for i in graph.inputs() {
        if out.len() >= 5 || i.identifier.starts_with('$') {
            continue;
        }
        let def: Vec<f64> = i.default.as_deref().unwrap_or("").split(',').filter_map(|s| s.trim().parse().ok()).collect();
        let pick = |k: usize, t: f64| -> f64 {
            let lo = i.min.as_ref().and_then(|m| m.get(k).or(m.first()).copied()).unwrap_or(0.0);
            let hi = i.max.as_ref().and_then(|m| m.get(k).or(m.first()).copied()).unwrap_or(1.0);
            lo + t * (hi - lo)
        };
        let (v, back) = match i.kind {
            InputKind::Float(n) if n >= 1 => {
                let mut v: Vec<f32> = (0..n).map(|k| pick(k, 0.3) as f32).collect();
                if v.iter().zip(&def).all(|(a, b)| (*a as f64 - b).abs() < 1e-6) {
                    v = (0..n).map(|k| pick(k, 0.7) as f32).collect();
                }
                let back: Vec<f32> = (0..n).map(|k| def.get(k).copied().unwrap_or(0.0) as f32).collect();
                (Value::Float(v), Value::Float(back))
            }
            InputKind::Int(n) if n >= 1 => {
                let d = def.first().copied().unwrap_or(0.0) as i32;
                let x = if let Some((o, _)) = i.options.iter().find(|(o, _)| *o != d) {
                    *o
                } else {
                    let m = pick(0, 0.5).round() as i32;
                    if m != d { m } else { d + 1 }
                };
                let back: Vec<i32> = (0..n).map(|k| def.get(k).copied().unwrap_or(0.0) as i32).collect();
                let mut v = back.clone();
                v[0] = x;
                (Value::Int(v), Value::Int(back))
            }
            _ => continue,
        };
        if first.is_none() {
            first = Some((i.identifier.clone(), back));
        }
        out.push((i.identifier.clone(), v));
    }
    out.push(("$randomseed".into(), Value::Int(vec![5])));
    out.push(("$outputsize".into(), Value::Int(vec![size + 1, size])));
    out.push(("$outputsize".into(), Value::Int(vec![size, size])));
    out.extend(first);
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let gpu = args[0] == "gpu";
    let size: i32 = args[1].parse().expect("LOG2SIZE");
    let (mut bad, mut steps) = (0, 0);
    for path in &args[2..] {
        let package = match Package::open(path) {
            Ok(p) => p,
            Err(e) => {
                println!("{path}: cannot open: {e}");
                continue;
            }
        };
        let graph = &package.graphs()[0];
        let (mut inc_r, mut full_r) = (renderer(gpu), renderer(gpu));
        let mut inst = Instance::new(graph);
        // INCR_BUDGET_MB: a small cache budget exercises eviction
        if let Some(mb) = std::env::var("INCR_BUDGET_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
            inst.set_cache_budget(mb << 20);
        }
        let mut params = Parameters::new();
        params.set("$outputsize", Value::Int(vec![size, size]));
        if let Err(e) = inst.render(&mut inc_r, graph, &params, None) {
            println!("{path}: render failed: {e}");
            continue;
        }
        let first = inst.last_stats();
        let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy().to_string();
        println!("{name}: first render {} nodes", first.recomputed);
        for (id, v) in changes(graph, size) {
            params.set(&id, v.clone());
            let t = Instant::now();
            let a = inst.render(&mut inc_r, graph, &params, None);
            let ti = t.elapsed().as_secs_f64();
            let st = inst.last_stats();
            let t = Instant::now();
            let b = full_r.render(graph, &params, None);
            let tf = t.elapsed().as_secs_f64();
            steps += 1;
            let same = match (&a, &b) {
                (Ok(a), Ok(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.identifier == y.identifier && x.width == y.width && x.height == y.height && x.data == y.data),
                (Err(x), Err(y)) => x.0 == y.0,
                _ => false,
            };
            if !same {
                bad += 1;
                match (&a, &b) {
                    (Err(e), _) => println!("  incremental render failed: {e}"),
                    (_, Err(e)) => println!("  full render failed: {e}"),
                    _ => println!("  outputs differ"),
                }
            }
            println!("  {} {id} = {v:?}: recomputed {} reused {} | incremental {ti:.3} s, full {tf:.3} s", if same { "ok      " } else { "MISMATCH" }, st.recomputed, st.reused);
        }
    }
    println!("{} backend: {steps} steps, {bad} mismatches", if gpu { "gpu" } else { "cpu" });
    std::process::exit(if bad > 0 { 1 } else { 0 });
}
