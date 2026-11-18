//! Research aid: a node's record and the instructions of its pixel-processor function; with
//! PIXEL=x,y also every register's value at that pixel (the size function runs first, as in a
//! render, for the function variables it sets).
//! cargo run --release --example dump_fn -- FILE.sbsar|.sbsasm NODE
use open_sbsar::program::{Program, Slot};
use open_sbsar::vm::{Env, Eval, Func, Value};

struct E {
    size: [f32; 2],
    pos: [f32; 2],
}
impl Env for E {
    fn input(&self, _uid: u32) -> Value {
        Value::int(&[0])
    }
    fn sysvar(&self, i: u16) -> Value {
        match i {
            1 => Value::float(&self.size),
            3 => Value::float(&[self.size[0].log2(), self.size[1].log2()]),
            8 => Value::float(&self.pos),
            _ => Value::f1(0.0),
        }
    }
    fn sample(&mut self, _: usize, _: [f32; 2], _: bool, _: bool) -> [f32; 4] {
        [0.0; 4]
    }
    fn vp_value(&self, _: usize) -> Value {
        Value::f1(0.0)
    }
    fn rand(&mut self) -> f32 {
        0.5
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let bytes = std::fs::read(&a[0]).unwrap();
    let asm = if a[0].ends_with(".sbsasm") { bytes } else { open_sbsar::package::read_archive(&bytes).unwrap().remove(0).asm };
    let p = Program::parse(asm).unwrap();
    if a[1] == "outputs" {
        for o in &p.outputs { println!("uid {} node {}", o.uid, o.node); }
        return;
    }
    let k: usize = a[1].parse().unwrap();
    let n = &p.nodes[k];
    let f = Func::decode(&p, n.data);
    let Ok(px) = std::env::var("PIXEL") else {
        println!("{n:#?}");
        for (i, ins) in f.ins.iter().enumerate() {
            println!("{i:4}: op {:2} ty {:2} ops {:?}", ins.op, ins.ty, &ins.ops[..]);
        }
        return;
    };
    let xy: Vec<f32> = px.split(',').map(|s| s.parse().unwrap()).collect();
    let (w, h) = n.size();
    let mut env = E { size: [w as f32, h as f32], pos: [(xy[0] + 0.5) / w as f32, (xy[1] + 0.5) / h as f32] };
    let mut vars = vec![];
    if let Slot::Fn(o) = n.size {
        let sf = Func::decode(&p, o);
        let v = Eval::run_with_vars(&sf, &mut env, &mut vars);
        println!("size fn -> {v:?}, vars {vars:?}");
    }
    for i in 0..f.ins.len() {
        let mut g = f.clone();
        g.ins.truncate(i + 1);
        let mut vv = vars.clone();
        println!("{i:4}: op {:2} ty {:2} ops {:?} = {:?}", f.ins[i].op, f.ins[i].ty, &f.ins[i].ops[..], Eval::run_with_vars(&g, &mut env, &mut vv));
    }
}
