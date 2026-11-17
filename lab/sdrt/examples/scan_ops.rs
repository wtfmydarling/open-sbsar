//! Static coverage scan: node opcodes and function instructions the engine does not implement.
//! cargo run --release --example scan_ops -- FILE.sbsar...
use open_sbsar::program::{Program, Slot};
use open_sbsar::vm::Func;

const NODE_OPS: &[u32] = &[0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2A, 0x2C];

fn vm_ok(op: u8) -> bool {
    !matches!(op, 5 | 8 | 10 | 14 | 25) && op <= 54
}

fn main() {
    for path in std::env::args().skip(1) {
        let name = std::path::Path::new(&path).file_name().unwrap().to_string_lossy().to_string();
        let bytes = match std::fs::read(&path) { Ok(b) => b, Err(e) => { println!("{name}: read error {e}"); continue } };
        let graphs = match open_sbsar::package::read_archive(&bytes) { Ok(g) => g, Err(e) => { println!("{name}: archive error {e}"); continue } };
        let (mut nodes, mut funcs, mut ins) = (0usize, 0usize, 0usize);
        let mut bad: Vec<String> = vec![];
        for g in graphs {
            let p = match Program::parse(g.asm) { Ok(p) => p, Err(e) => { bad.push(format!("parse error {e}")); continue } };
            let mut offs = vec![];
            for n in &p.nodes {
                nodes += 1;
                let op = n.op & 0xFE;
                if !NODE_OPS.contains(&op) { bad.push(format!("node {} op {:#04x}", n.index, n.op)); }
                for s in [&n.seed, &n.size, &n.pixelsize].into_iter().chain(n.params.iter().map(|(_, s)| s)) {
                    if let Slot::Fn(o) = s { offs.push(*o); }
                }
                if op == 0x28 { offs.push(n.data); }
            }
            offs.sort_unstable();
            offs.dedup();
            for o in offs {
                let f = Func::decode(&p, o);
                funcs += 1;
                for i in &f.ins {
                    ins += 1;
                    if !vm_ok(i.op) { bad.push(format!("fn {o} vm op {}", i.op)); }
                }
            }
        }
        bad.sort();
        bad.dedup();
        println!("{name}: nodes {nodes} functions {funcs} instructions {ins} unknown {} {:?}", bad.len(), &bad[..bad.len().min(6)]);
    }
}
