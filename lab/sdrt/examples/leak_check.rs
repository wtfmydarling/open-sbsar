//! Memory over repeated renders: RSS (and with the GPU, this process's device memory from
//! nvidia-smi) after each render on one renderer, plain and through an Instance.
//! cargo run --release --example leak_check -- cpu|gpu LOG2SIZE ROUNDS FILE.sbsar...
use open_sbsar::{Instance, Package, Parameters, Renderer, Value};

fn rss_mb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) / 1024
}

fn vram_mb() -> u64 {
    let out = std::process::Command::new("nvidia-smi").args(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).output();
    let me = std::process::id().to_string();
    out.ok().map(|o| String::from_utf8_lossy(&o.stdout).lines().filter_map(|l| {
        let mut it = l.split(',').map(str::trim);
        (it.next() == Some(&me)).then(|| it.next().and_then(|v| v.parse::<u64>().ok())).flatten()
    }).sum()).unwrap_or(0)
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let gpu = a[0] == "gpu";
    let size: i32 = a[1].parse().unwrap();
    let rounds: usize = a[2].parse().unwrap();
    let packages: Vec<Package> = a[3..].iter().map(|p| Package::open(p).expect("open")).collect();
    let mut r = if gpu { Renderer::new_gpu().expect("gpu") } else { Renderer::new_cpu() };
    let mut insts: Vec<Instance> = packages.iter().map(|p| Instance::new(&p.graphs()[0])).collect();
    for round in 0..rounds {
        for (k, p) in packages.iter().enumerate() {
            let g = &p.graphs()[0];
            let mut params = Parameters::new();
            // alternate sizes and seeds: resolution changes reallocate everything
            params.set("$outputsize", Value::Int(vec![size - (round % 3) as i32, size - (round % 3) as i32]));
            params.set("$randomseed", Value::Int(vec![(round % 5) as i32]));
            r.render(g, &params, None).expect("render");
            insts[k].render(&mut r, g, &params, None).expect("instance render");
        }
        println!("round {round:3} rss {:6} MB vram {:6} MB", rss_mb(), if gpu { vram_mb() } else { 0 });
    }
    drop(insts);
    drop(r);
    println!("after drop rss {:6} MB vram {:6} MB", rss_mb(), if gpu { vram_mb() } else { 0 });
}
