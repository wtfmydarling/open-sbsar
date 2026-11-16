//! A compiled program (.sbsasm): header, interface, input defaults and node records, laid out as
//! research/sbsar/NOTES.md describes. Records are parsed into named parameter slots.

use std::collections::HashMap;

use crate::vm::Value;

/// A parameter: absent (the op's default), a constant inline, a function, or an image input.
#[derive(Clone, Debug)]
pub enum Slot {
    None,
    Const(Vec<u32>),
    Fn(usize),
    Input(usize),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub index: usize,
    pub off: usize,
    pub op: u32,
    pub color: bool,
    pub log2: (u32, u32),
    pub b2: u32,
    pub b3: u32,
    pub flags: u32,
    pub inputs: Vec<usize>,
    pub seed: Slot,
    pub size: Slot,
    /// block offset of the size slot's word (a function's offset), when there is one
    pub size_at: usize,
    pub pixelsize: Slot,
    pub params: Vec<(&'static str, Slot)>,
    /// Block offset of op-specific data (gradient keys, curves, bitmap, SVG stream, string).
    pub data: usize,
    pub data2: usize,
    pub count: u32,
}

impl Node {
    pub fn param(&self, name: &str) -> &Slot {
        self.params.iter().find(|(n, _)| *n == name).map(|(_, s)| s).unwrap_or(&Slot::None)
    }
    /// 0 8-bit, 1 16-bit, 2 16F, 3 32F.
    pub fn format(&self) -> u32 {
        (self.b2 >> 4) & 3
    }
    pub fn size(&self) -> (usize, usize) {
        (1 << self.log2.0.min(13), 1 << self.log2.1.min(13))
    }
}

pub struct Output {
    pub uid: u32,
    pub flags: u32,
    pub node: usize,
}

pub struct Program {
    pub b: Vec<u8>,
    pub base: usize,
    pub version: u32,
    pub build: u32,
    pub nodes: Vec<Node>,
    pub outputs: Vec<Output>,
    /// (type, uid) of each input, sorted by uid.
    pub inputs: Vec<(u32, u32)>,
    pub defaults: HashMap<u32, Value>,
}

fn words_of_type(t: u32, version: u32) -> usize {
    match t {
        0 | 4 => 1,
        1 | 8 => 2,
        2 | 9 => 3,
        3 | 10 => 4,
        5 if version >= 8 => 4,
        _ => 0,
    }
}

impl Program {
    pub fn w(&self, off: usize) -> u32 {
        let o = self.base + off;
        if o + 4 <= self.b.len() {
            u32::from_le_bytes(self.b[o..o + 4].try_into().unwrap())
        } else {
            0
        }
    }
    pub fn f(&self, off: usize) -> f32 {
        f32::from_bits(self.w(off))
    }
    pub fn bytes(&self, off: usize, n: usize) -> &[u8] {
        let start = (self.base + off).min(self.b.len());
        let end = (start + n).min(self.b.len());
        &self.b[start..end]
    }

    pub fn parse(b: Vec<u8>) -> Result<Program, String> {
        let u = |o: usize| if o + 4 <= b.len() { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) } else { 0 };
        if b.len() < 0x40 || &b[..4] != b"SBAM" {
            return Err("not an .sbsasm".into());
        }
        let version = (u(4) >> 16) & 0xFF;
        let build = u(8);
        let p = u(0x2c) as usize;
        let base = 0x34;
        let iface = base + p;
        if iface + 4 > b.len() {
            return Err("truncated interface".into());
        }
        let n_out = u16::from_le_bytes([b[iface], b[iface + 1]]) as usize;
        let n_in = u16::from_le_bytes([b[iface + 2], b[iface + 3]]) as usize;
        if n_out > 10_000 || n_in > 10_000 || iface + 4 + 4 * n_out > b.len() {
            return Err("invalid io count".into());
        }
        let uids: Vec<u32> = (0..n_out).map(|k| u(iface + 4 + 4 * k)).collect();
        let io = iface + 4 + 4 * n_out;
        if io + 8 * n_in > b.len() {
            return Err("truncated inputs".into());
        }
        let inputs: Vec<(u32, u32)> = (0..n_in).map(|k| (u(io + 8 * k), u(io + 8 * k + 4))).collect();
        let tr = io + 8 * n_in;
        if tr + 16 > b.len() {
            return Err("truncated table roots".into());
        }
        let (n_nodes, node_table, out_table, defaults) = (u(tr) as usize, u(tr + 4) as usize, u(tr + 8) as usize, u(tr + 12) as usize);
        if n_nodes > b.len() / 4 {
            return Err("node count larger than the program".into());
        }
        let mut prog = Program { b, base, version, build, nodes: vec![], outputs: vec![], inputs, defaults: HashMap::new() };
        prog.outputs = (0..n_out)
            .map(|k| Output { uid: uids[k], flags: prog.w(out_table + 8 * k), node: prog.w(out_table + 8 * k + 4) as usize })
            .collect();
        if defaults != 0 {
            let mut o = defaults;
            for &(t, uid) in &prog.inputs.clone() {
                let n = words_of_type(t, version);
                let ws: Vec<u32> = (0..n).map(|i| prog.w(o + 4 * i)).collect();
                o += 4 * n;
                let v = match t {
                    0..=3 => Value::float(&ws.iter().map(|&w| f32::from_bits(w)).collect::<Vec<_>>()),
                    4 | 8..=10 => Value::int(&ws.iter().map(|&w| w as i32).collect::<Vec<_>>()),
                    5 if n == 4 => Value::float(&ws.iter().map(|&w| f32::from_bits(w)).collect::<Vec<_>>()),
                    _ => continue,
                };
                prog.defaults.insert(uid, v);
            }
        }
        let offs: Vec<usize> = (0..n_nodes).map(|i| prog.w(node_table + 4 * i) as usize).collect();
        for (i, &off) in offs.iter().enumerate() {
            let node = parse_node(&prog, i, off)?;
            prog.nodes.push(node);
        }
        Ok(prog)
    }
}

/// Reads the words of a record one after another.
struct Cur<'a> {
    p: &'a Program,
    o: usize,
}

impl Cur<'_> {
    fn word(&mut self) -> u32 {
        let v = self.p.w(self.o);
        self.o += 4;
        v
    }
    fn words(&mut self, n: usize) -> Vec<u32> {
        // words past the end read as 0; a count beyond the program is malformed
        let n = n.min(self.p.b.len() / 4);
        (0..n).map(|_| self.word()).collect()
    }
    fn slot(&mut self, bits: u32, n: usize) -> Slot {
        match bits & 3 {
            1 => Slot::Const(self.words(n)),
            2 => Slot::Fn(self.word() as usize),
            3 => Slot::Input(self.word() as usize),
            _ => Slot::None,
        }
    }
}

fn parse_node(p: &Program, index: usize, off: usize) -> Result<Node, String> {
    let opw = p.w(off);
    let code = opw & 0xFE;
    let color = opw & 1 == 1;
    let b2 = (opw >> 16) & 0xFF;
    let b3 = opw >> 24;
    let mut n = Node {
        index,
        off,
        op: code,
        color,
        log2: ((opw >> 8) & 0xF, (opw >> 12) & 0xF),
        b2,
        b3,
        flags: 0,
        inputs: vec![],
        seed: Slot::None,
        size: Slot::None,
        size_at: 0,
        pixelsize: Slot::None,
        params: vec![],
        data: 0,
        data2: 0,
        count: 0,
    };
    let mut c = Cur { p, o: off + 4 };
    let nc = if color { 4 } else { 1 };
    macro_rules! common {
        () => {{
            n.seed = c.slot(b2 >> 6, 1);
            if b2 & 1 == 1 { n.size_at = c.o; }
            n.size = if b2 & 1 == 1 { c.slot(2, 1) } else { Slot::None };
        }};
    }
    macro_rules! param {
        ($name:expr, $bits:expr, $nw:expr) => {{
            let s = c.slot($bits, $nw);
            n.params.push(($name, s));
        }};
    }
    match code {
        0x00 => {
            // gradient map
            n.inputs.push(c.word() as usize);
            n.count = c.word().min(p.b.len() as u32 / 4);
            if p.version >= 8 && b3 & 2 != 0 {
                n.params.push(("v8word", Slot::Const(vec![c.word()])));
            }
            n.data = c.word() as usize;
            common!();
        }
        0x02 => {
            // blend: flags, background, foreground
            n.flags = c.word();
            let bg = c.word() as usize;
            let fg = c.word() as usize;
            n.inputs = vec![bg, fg];
            common!();
            param!("opacity", n.flags >> 4, 1);
            if (n.flags >> 4) & 3 == 3 {
                param!("opacitymult", n.flags >> 9, 1);
            }
            if n.flags & 0x40 != 0 {
                n.params.push(("mask", Slot::Const(c.words(2))));
            }
        }
        0x04 => {
            n.flags = c.word();
            n.inputs.push(c.word() as usize);
            common!();
            if n.flags & 0x1F == 0x1E {
                param!("miplevel", 2, 1);
            }
            param!("matrix", n.flags >> 6, 4);
            param!("offset", n.flags >> 25, 2);
            param!("mattecolor", n.flags >> 28, 4);
        }
        0x06 if !color => {
            n.inputs.push(c.word() as usize);
            common!();
            param!("weights", b3, 4);
            param!("mattelevel", b3 >> 2, 1);
        }
        0x06 => {
            n.flags = c.word();
            let a = c.word() as usize;
            let b = c.word() as usize;
            n.inputs = vec![a, b];
            common!();
        }
        0x08 => {
            // FX-Map: walked by fxmap.rs from the record offset; inputs and slots read here
            n.flags = c.word();
            n.data = c.word() as usize; // tree
            let nin = ((n.flags >> 10) & 0x3F) as usize;
            n.inputs = c.words(nin).into_iter().map(|w| w as usize).collect();
            common!();
            if n.flags & 3 == 3 {
                n.params.push(("background", Slot::Input(c.word() as usize)));
            } else {
                param!("background", n.flags, nc);
            }
            param!("hurst", n.flags >> 2, 1);
            param!("opacitymult", n.flags >> 4, 1);
            param!("renderroi", n.flags >> 6, 4);
            param!("tilingroi", n.flags >> 8, 4);
        }
        0x0A => {
            n.data = c.word() as usize;
            if b3 & 4 != 0 {
                n.inputs.push(c.word() as usize);
            }
            common!();
            if b3 & 2 != 0 {
                n.params.push(("color", Slot::Const(c.words(nc))));
            }
        }
        0x0C => {
            common!();
            param!("color", b3, nc);
        }
        0x0E => {
            if p.version >= 9 {
                n.flags = c.word();
            }
            let a = c.word() as usize;
            let g = c.word() as usize;
            n.inputs = vec![a, g];
            common!();
            n.pixelsize = c.slot(b3 >> 2, 2);
            param!("intensity", b3 >> 5, 1);
        }
        0x10 => {
            n.flags = c.word();
            let a = c.word() as usize;
            let g = c.word() as usize;
            n.inputs = vec![a, g];
            common!();
            n.pixelsize = c.slot(b3 >> 2, 2);
            param!("intensity", n.flags >> 1, 1);
            param!("lightangle", n.flags >> 3, 1);
            param!("highlight", n.flags >> 5, 4);
            param!("shadow", n.flags >> 7, 4);
        }
        0x12 => {
            n.flags = c.word();
            let a = c.word() as usize;
            let g = c.word() as usize;
            n.inputs = vec![a, g];
            common!();
            n.pixelsize = c.slot(b3 >> 2, 2);
            param!("intensity", n.flags, 1);
            param!("shiftangle", n.flags >> 2, 1);
        }
        0x14 | 0x1A => {
            n.inputs.push(c.word() as usize);
            common!();
            n.pixelsize = c.slot(b3 >> 2, 2);
            param!("intensity", b3 >> 4, 1);
        }
        0x16 => {
            n.flags = c.word();
            n.inputs.push(c.word() as usize);
            common!();
            n.pixelsize = c.slot(b3 >> 2, 2);
            param!("intensity", n.flags, 1);
            param!("angle", n.flags >> 2, 1);
        }
        0x18 => {
            n.flags = c.word();
            let a = c.word() as usize;
            let i = c.word() as usize;
            n.inputs = vec![a, i];
            common!();
            n.pixelsize = c.slot(b3 >> 2, 2);
            param!("intensity", n.flags >> 1, 1);
            param!("angle", n.flags >> 3, 1);
            param!("v9third", n.flags >> 7, 1);
        }
        0x1C => {
            n.inputs.push(c.word() as usize);
            common!();
            param!("hue", b3, 1);
            param!("saturation", b3 >> 2, 1);
            param!("luminosity", b3 >> 4, 1);
        }
        0x1E => {
            n.flags = c.word();
            n.inputs.push(c.word() as usize);
            common!();
            for (name, bit) in [("inlow", 0), ("inhigh", 2), ("inmid", 4), ("outlow", 6), ("outhigh", 8)] {
                param!(name, n.flags >> bit, nc);
            }
        }
        0x20 => {
            n.data = c.word() as usize; // input uid, or block offset of the pixels
            if b3 & 8 != 0 && b3 & 1 != 0 {
                n.data2 = c.word() as usize;
            }
            common!();
        }
        0x22 => {
            n.flags = c.word();
            c.word();
            n.data = c.word() as usize; // string offset, or input uid
            n.data2 = c.word() as usize; // font offset
            common!();
            // a background input's index comes before the parameters (petal_embroidery 92: read
            // last, it shifted every parameter by one word)
            if (n.flags >> 2) & 3 == 3 {
                n.inputs.push(c.word() as usize);
            }
            // order measured on the corpus records: matrix before position (car_air_freshener 171:
            // a rotation-scale, then (0.19, −0.016), then fontsize 0.11); colour's place is a guess
            for (name, bit, nw) in [("color", 4, nc), ("matrix", 10, 4), ("position", 6, 2), ("fontsize", 8, 1)] {
                param!(name, n.flags >> bit, nw);
            }
            if (n.flags >> 12) & 3 == 3 {
                param!("alignment", 2, 1);
            }
        }
        0x24 => {
            n.flags = c.word();
            n.inputs.push(c.word() as usize);
            common!();
            n.pixelsize = c.slot(b3 >> 2, 2);
            param!("intensity", n.flags, 1);
            // bool slots: 1 = true with no word after it, 2 = a function
            for (name, bits) in [("inversedy", n.flags >> 2), ("input2alpha", n.flags >> 4)] {
                let s = match bits & 3 {
                    1 => Slot::Const(vec![1]),
                    2 => c.slot(2, 1),
                    _ => Slot::None,
                };
                n.params.push((name, s));
            }
        }
        0x26 => {
            let a = c.word() as usize;
            let b = c.word() as usize;
            n.inputs = vec![a, b];
            common!();
            param!("coordinate", b3 >> 1, 1);
        }
        0x28 => {
            n.flags = c.word();
            let k = (n.flags & 0xFFFF) as usize;
            n.inputs = c.words(k).into_iter().map(|w| w as usize).collect();
            common!();
            n.data = c.word() as usize; // the function
        }
        0x2A => {
            n.flags = c.word();
            n.inputs.push(c.word() as usize);
            if n.flags & 1 != 0 {
                n.inputs.push(c.word() as usize);
            }
            common!();
            n.pixelsize = c.slot(b3 >> 2, 2);
            param!("distance", n.flags >> 2, 1);
            if n.flags & 0x40 != 0 {
                param!("mode", 2, 1);
            }
        }
        0x2C => {
            n.inputs.push(c.word() as usize);
            n.count = c.word().min(p.b.len() as u32 / 4);
            n.data = c.word() as usize;
            common!();
        }
        _ => return Err(format!("node {index}: unknown op {code:#04x}")),
    }
    Ok(n)
}
