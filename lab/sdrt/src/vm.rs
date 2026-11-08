//! Functions (dynamic values): decoding and evaluation, as research/sbsar/NOTES.md describes.
//!
//! Evaluation is lazy, memoized and per component, as measured with sbsrender: the result's
//! components are computed in order, each one demanding only the operand components it needs
//! (operands in order), every (register, component) is computed once. An ifelse visits its true and
//! false branches in that order: the chosen branch runs normally, while the other contributes only
//! shared expressions without a direct write. Sequence operands with side effects run whole.
//! Random draws happen in that order.

use crate::program::Program;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Void,
    B(bool),
    F(u8, [f32; 4]),
    I(u8, [i32; 4]),
    /// A string input (its uid).
    S(u32),
}

impl Value {
    pub fn float(v: &[f32]) -> Value {
        let mut a = [0.0; 4];
        a[..v.len()].copy_from_slice(v);
        Value::F(v.len() as u8, a)
    }
    pub fn int(v: &[i32]) -> Value {
        let mut a = [0; 4];
        a[..v.len()].copy_from_slice(v);
        Value::I(v.len() as u8, a)
    }
    pub fn f1(x: f32) -> Value {
        Value::F(1, [x, 0.0, 0.0, 0.0])
    }
    pub fn n(&self) -> usize {
        match self {
            Value::F(n, _) | Value::I(n, _) => *n as usize,
            Value::B(_) => 1,
            _ => 0,
        }
    }
    /// Components as floats (ints converted, bool 0/1).
    pub fn fv(&self) -> [f32; 4] {
        match *self {
            Value::F(_, a) => a,
            Value::I(_, a) => a.map(|x| x as f32),
            Value::B(b) => [b as i32 as f32; 4],
            _ => [0.0; 4],
        }
    }
    pub fn iv(&self) -> [i32; 4] {
        match *self {
            Value::I(_, a) => a,
            Value::F(_, a) => a.map(|x| x as i32),
            Value::B(b) => [b as i32; 4],
            _ => [0; 4],
        }
    }
    pub fn as_f(&self) -> f32 {
        self.fv()[0]
    }
    pub fn as_bool(&self) -> bool {
        match *self {
            Value::B(b) => b,
            Value::F(_, a) => a[0] != 0.0,
            Value::I(_, a) => a[0] != 0,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Ins {
    pub op: u8,
    pub ty: u8,
    pub ops: Ops,
}

/// An instruction's operands, inline up to 10 (one contiguous instruction array: big pixel
/// processors have tens of thousands of instructions), longer lists on the heap.
#[derive(Clone)]
pub struct Ops {
    len: u8,
    a: [u16; 10],
    spill: Option<Box<[u16]>>,
}

impl From<&[u16]> for Ops {
    fn from(v: &[u16]) -> Ops {
        if v.len() <= 10 {
            let mut a = [0; 10];
            a[..v.len()].copy_from_slice(v);
            Ops { len: v.len() as u8, a, spill: None }
        } else {
            Ops { len: 0, a: [0; 10], spill: Some(v.into()) }
        }
    }
}

impl std::ops::Deref for Ops {
    type Target = [u16];
    fn deref(&self) -> &[u16] {
        match &self.spill {
            Some(v) => v,
            None => &self.a[..self.len as usize],
        }
    }
}

impl std::fmt::Debug for Ops {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

#[derive(Clone, Debug)]
pub struct Func {
    pub ins: Vec<Ins>,
    side: Vec<bool>,
    /// Per register component: live with two or more consumers. Such a component is also
    /// materialized when encountered in an untaken branch, in expression traversal order
    /// (lab/py/ifelse_order.py). Single-use expressions stay lazy. A shared setvar/write/loop
    /// itself is excluded from this extra evaluation. Liveness and consumers are counted per
    /// scalar component, from the components each instruction demands of its operands: with
    /// ifelse(c, v, swizzle(v, 1)), v.y is shared when both result components are demanded, and
    /// single-use (stays lazy) when only one is (scratch/fb/partprobe.py, boilingground node 17).
    hoist: Vec<[bool; 4]>,
    /// Literal words of constants (op 0) and inputs (op 2), decoded once.
    lits: Vec<[u32; 4]>,
    /// per loop condition / body register: the registers each iteration recomputes
    loop_inv: Vec<(usize, Vec<usize>)>,
    pub off: usize,
}

/// Component c of register k, clamped as `Eval::comp` does.
fn clamp_comp(ins: &[Ins], k: usize, c: usize) -> usize {
    if comps(ins[k].ty) == 1 { 0 } else { c.min(3) }
}

/// The operand components (register, component) that computing component c of instruction k reads,
/// in evaluation order (mirrors `Eval::compute`).
fn scalar_reads(ins: &[Ins], side: &[bool], k: usize, c: usize, out: &mut Vec<(usize, usize)>) {
    let i = &ins[k];
    let n = ins.len();
    let op = |j: usize| i.ops.get(j).map_or(usize::MAX, |&x| x as usize);
    let mut push = |r: usize, rc: usize| {
        if r < k && r < n {
            out.push((r, clamp_comp(ins, r, rc)));
        }
    };
    let all = |r: usize| if r < n { comps(ins[r].ty) } else { 0 };
    match i.op {
        0..=4 => {}
        6 => (0..all(op(0))).for_each(|j| push(op(0), j)),
        7 | 17 | 23 | 35..=44 | 53 | 50 => push(op(0), c),
        9 => {
            push(op(0), 0);
            push(op(1), c);
            push(op(2), c);
        }
        11 => {
            (0..all(op(0))).for_each(|j| push(op(0), j));
            push(op(1), 0);
            (0..all(op(2))).for_each(|j| push(op(2), j));
        }
        12 => {
            // the first operand counts as a consumer even without side effects, when the sequence
            // does not run it (multiwood node 64: seq(rand, rand') keeps rand' shared)
            let (a, b) = (op(0), op(1));
            (0..all(a)).for_each(|j| push(a, j));
            if side.get(b).copied().unwrap_or(false) {
                (0..all(b)).for_each(|j| push(b, j));
            }
            push(b, c);
        }
        13 => {
            let na = all(op(0));
            if c < na { push(op(0), c) } else { push(op(1), c - na) }
        }
        15 => push(op(c.min(3)), 0),
        16 => push(op(0), ((i.ops[1] >> (2 * c)) & 3) as usize),
        18..=22 | 48 | 49 | 54 => {
            for r in [op(0), op(1)] {
                let zero_ext = i.op != 20 && c > 0 && r < n && comps(ins[r].ty) == 1 && comps(i.ty) > 1;
                if !zero_ext {
                    push(r, c);
                }
            }
        }
        24 => {
            let m = all(op(0)).min(all(op(1))).max(1);
            for j in 0..m {
                push(op(0), j);
                push(op(1), j);
            }
        }
        26 | 27 | 29..=34 | 46 => {
            push(op(0), 0);
            push(op(1), 0);
        }
        28 => push(op(0), 0),
        45 | 51 | 52 => {
            push(op(0), 0);
            push(op(0), 1);
        }
        47 => {
            push(op(0), c);
            push(op(1), c);
            push(op(2), c);
        }
        _ => {}
    }
}

/// Which operands of an instruction are registers.
fn reg_operands(i: &Ins) -> &[u16] {
    let n = match i.op {
        0..=4 => 0,
        6 | 7 | 16 | 17 | 51 | 52 => 1,
        9 | 11 | 47 => 3,
        15 => 4,
        23 | 28 | 35..=45 | 50 | 53 => 1,
        _ => 2,
    };
    &i.ops[..n.min(i.ops.len())]
}

impl Func {
    pub fn decode(p: &Program, off: usize) -> Func {
        let b = &p.b;
        let mut o = p.base + off;
        let rd = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let n = rd(o) as usize;
        o += 2;
        let mut ins = Vec::with_capacity(n);
        for _ in 0..n {
            let h = rd(o);
            let k = (h >> 10) as usize;
            let ops: Vec<u16> = (0..k).map(|j| rd(o + 2 + 2 * j)).collect();
            ins.push(Ins { op: (h & 0x3F) as u8, ty: ((h >> 6) & 0xF) as u8, ops: Ops::from(&ops[..]) });
            o += 2 + 2 * k;
        }
        let mut side = vec![false; ins.len()];
        for (k, i) in ins.iter().enumerate() {
            side[k] = matches!(i.op, 6 | 7 | 11) || reg_operands(i).iter().any(|&r| (r as usize) < k && side[r as usize]);
        }
        let n = ins.len();
        // demanded components, from the result's (all of them) back to the operands
        let mut need = vec![0u8; n];
        if n > 0 {
            need[n - 1] = ((1u32 << comps(ins[n - 1].ty).min(4)) - 1) as u8;
        }
        // consumers per register component: distinct scalar consumers (register, component), or
        // with SDRT_SHARE=reg distinct consumer registers (test switch)
        let per_reg = crate::env_once!("SDRT_SHARE").as_deref() == Ok("reg");
        let mut users = vec![[0u32; 4]; n];
        let mut last_user = vec![[usize::MAX; 4]; n];
        let mut reads = Vec::new();
        for k in (0..n).rev() {
            for c in 0..4 {
                if need[k] & (1 << c) == 0 {
                    continue;
                }
                reads.clear();
                scalar_reads(&ins, &side, k, c, &mut reads);
                let mut seen: Vec<(usize, usize)> = vec![];
                for &(r, rc) in &reads {
                    need[r] |= 1 << rc;
                    if seen.contains(&(r, rc)) {
                        continue;
                    }
                    seen.push((r, rc));
                    let id = if per_reg { k } else { k * 4 + c };
                    if last_user[r][rc] != id {
                        last_user[r][rc] = id;
                        users[r][rc] += 1;
                    }
                }
            }
        }
        let hoist = (0..n).map(|k| std::array::from_fn(|c| need[k] & (1 << c) != 0 && users[k][c] >= 2)).collect();
        let lits = ins
            .iter()
            .map(|i| {
                let start = i.ops.len() % 2;
                let word = |w: usize| match (i.ops.get(start + 2 * w), i.ops.get(start + 2 * w + 1)) {
                    (Some(&lo), Some(&hi)) => lo as u32 | (hi as u32) << 16,
                    _ => 0,
                };
                if matches!(i.op, 0 | 2) && !(i.op == 0 && i.ty == 1) { std::array::from_fn(word) } else { [0; 4] }
            })
            .collect();
        // registers whose value can change between loop iterations: variable reads/writes, writes,
        // loops, rand, or anything computed from them (the rest is the same in every iteration)
        let mut volatile = vec![false; n];
        for (k, i) in ins.iter().enumerate() {
            volatile[k] = matches!(i.op, 4 | 6 | 7 | 11 | 50) || reg_operands(i).iter().any(|&r| (r as usize) < k && volatile[r as usize]);
        }
        // loops: the registers their condition and body recompute every iteration (SDRT_LOOPALL=1:
        // every dependency, old)
        let all = crate::env_once!("SDRT_LOOPALL").is_ok();
        let mut loop_inv = vec![];
        for i in ins.iter().filter(|i| i.op == 11) {
            for &root in i.ops.iter().skip(1).take(2) {
                let k = root as usize;
                if k >= n || loop_inv.iter().any(|e: &(usize, Vec<usize>)| e.0 == k) {
                    continue;
                }
                let mut stack = vec![k];
                let mut seen = vec![false; n];
                let mut list = vec![];
                while let Some(r) = stack.pop() {
                    if r >= n || seen[r] {
                        continue;
                    }
                    seen[r] = true;
                    list.push(r);
                    for &o in reg_operands(&ins[r]) {
                        let o = o as usize;
                        if o < r && (all || volatile[o]) {
                            stack.push(o);
                        }
                    }
                }
                loop_inv.push((k, list));
            }
        }
        Func { ins, side, hoist, lits, loop_inv, off }
    }

    /// GPU code generation reuses the decoder's liveness and volatility analysis.
    #[cfg(feature = "gpu")]
    pub fn gpu_side(&self) -> &[bool] { &self.side }

    #[cfg(feature = "gpu")]
    pub fn gpu_invalidate(&self, root:usize) -> Vec<usize> {
        self.loop_inv.iter().find(|e| e.0==root).map_or_else(||if root<self.ins.len() {vec![root]} else {vec![]},|e|e.1.clone())
    }

    #[cfg(feature = "gpu")]
    pub fn gpu_shared(&self, root:usize, c:usize) -> Vec<(usize,usize)> {
        let ins=&self.ins;
        let mut stack=if root<ins.len() {vec![(root,clamp_comp(ins,root,c))]} else {vec![]};
        let mut found=Vec::new(); let mut reads=Vec::new();
        while let Some((r,rc))=stack.pop() {
            if found.contains(&(r,rc)) {continue;}
            if self.hoist[r][rc] {
                if !matches!(ins[r].op,6|7|11) {found.push((r,rc));}
                continue;
            }
            reads.clear(); scalar_reads(ins,&self.side,r,rc,&mut reads);
            stack.extend(reads.iter().rev());
        }
        found
    }

    /// Whether component c of register k is materialized in an untaken branch (`hoist`).
    #[cfg(feature = "gpu")]
    pub fn gpu_hoisted(&self, k: usize, c: usize) -> bool {
        k < self.ins.len() && self.hoist[k][clamp_comp(&self.ins, k, c)]
    }

    /// The operand components computing component c of register k reads, in evaluation order.
    #[cfg(feature = "gpu")]
    pub fn gpu_reads(&self, k: usize, c: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if k < self.ins.len() {
            scalar_reads(&self.ins, &self.side, k, clamp_comp(&self.ins, k, c), &mut out);
        }
        out
    }

    /// Registers whose value can change between evaluations of the same pixel's function:
    /// variable reads/writes, writes, loops, rand, or anything computed from them (the
    /// decoder's `volatile`). The others have no side effects and depend on nothing stateful.
    #[cfg(feature = "gpu")]
    pub fn gpu_volatile(&self) -> Vec<bool> {
        let mut volatile = vec![false; self.ins.len()];
        for (k, i) in self.ins.iter().enumerate() {
            volatile[k] = matches!(i.op, 4 | 6 | 7 | 11 | 50) || reg_operands(i).iter().any(|&r| (r as usize) < k && volatile[r as usize]);
        }
        volatile
    }

    /// Whether register k names a register operand at or after itself (scalar_reads skips
    /// those; the evaluator would still compute them).
    #[cfg(feature = "gpu")]
    pub fn gpu_forward(&self, k: usize) -> bool {
        k < self.ins.len() && reg_operands(&self.ins[k]).iter().any(|&r| (r as usize) >= k && (r as usize) < self.ins.len())
    }

    /// Registers whose value is the same at every pixel of a pixel processor: no $pos, sampler,
    /// rand, variable, write or loop below them.
    pub fn pixel_invariant(&self) -> Vec<bool> {
        let mut inv = vec![false; self.ins.len()];
        for (k, i) in self.ins.iter().enumerate() {
            let own = match i.op {
                1 => i.ops.first() != Some(&8),
                4 | 6 | 7 | 11 | 50 | 51 | 52 => false,
                _ => true,
            };
            inv[k] = own && reg_operands(i).iter().all(|&r| (r as usize) < k && inv[r as usize]);
        }
        inv
    }

    /// Same result at every evaluation of a render: no rand, position/iteration/depth system
    /// variables, variables, writes, loops, samples or value-processor reads.
    pub fn pure(&self) -> bool {
        self.ins.iter().all(|i| match i.op {
            1 => !matches!(i.ops.first(), Some(8..=12)),
            3 | 4 | 6 | 7 | 11 | 50 | 51 | 52 => false,
            _ => true,
        })
    }

    /// The u32 an instruction carries in its operands (after the alignment pad, if any).
    fn literal(i: &Ins, word: usize) -> u32 {
        let start = i.ops.len() % 2;
        let lo = i.ops[start + 2 * word] as u32;
        let hi = i.ops[start + 2 * word + 1] as u32;
        lo | hi << 16
    }

    /// The type of a register's result: (base kind, components). Kinds: b bool, f float, i int.
    pub fn result_type(&self) -> u8 {
        self.ins.last().map_or(0, |i| i.ty)
    }
}

/// What a function can ask of the node evaluating it.
pub trait Env {
    fn input(&self, uid: u32) -> Value;
    fn sysvar(&self, index: u16) -> Value;
    fn sample(&mut self, input: usize, pos: [f32; 2], color: bool, nearest: bool) -> [f32; 4];
    /// The sampler instruction's raw filter operand, before `sample` (FX-Maps test its bit 1).
    fn sample_filter(&mut self, _filter: u16) {}
    fn vp_value(&self, node: usize) -> Value;
    /// The next uniform draw in [0, 1).
    fn rand(&mut self) -> f32;
    /// write(value, slot): a pixel processor's output, or a value processor's output slot `slot`
    fn write(&mut self, _slot: u16, _v: Value) {}
    /// A variable slot read or written (FX-Maps record them for the parallel traversal).
    fn var_access(&mut self, _slot: u16, _write: bool) {}
    /// Functions evaluated by Substance's CPU evaluator (FX-Map functions) use its own sine
    /// approximation (`sbs_sin`); pixel processors run on the GPU and use the hardware's.
    fn cpu_trig(&self) -> bool {
        false
    }
}

/// CPU function trig, measured through self-authored FX-Maps (scratch/fx_polish).
/// Fold turns to a signed triangle in [-1,1], then evaluate an odd polynomial in f32.
/// Both phase folding and the order of the separate polynomial terms matter: converting back
/// to radians and evaluating in f64 moves large scatter offsets by several f32 ulps.
pub fn sbs_trig_turns(u: f32, phase: f32) -> f32 {
    let u = u + phase;
    let u = (u % 1.0).abs();
    let u = if u > 0.5 { 1.0 - u } else { u };
    let x = u * 4.0 - 1.0;
    let x2 = x * x;
    let x3 = x * x2;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    x * f32::from_bits(0x3fc9_0fdb)
        + x3 * f32::from_bits(0xbf25_5de7)
        + x5 * f32::from_bits(0x3da3_35e3)
        + x7 * f32::from_bits(0xbb94_463d)
}

pub fn sbs_sin_turns(u: f32) -> f32 {
    sbs_trig_turns(u, 0.25)
}

pub fn sbs_sin(t: f32) -> f32 {
    sbs_sin_turns(t * (1.0 / std::f32::consts::TAU))
}

pub fn sbs_cos(t: f32) -> f32 {
    sbs_trig_turns(t * (1.0 / std::f32::consts::TAU), 0.5)
}

#[allow(dead_code)]
fn typed_float(ty: u8, v: [f32; 4]) -> Value {
    match ty {
        1 => Value::B(v[0] != 0.0),
        4..=7 => Value::F(ty - 3, v),
        8..=11 => Value::I(ty - 7, v.map(|x| x as i32)),
        _ => Value::F(1, v),
    }
}

/// One component of a register: functions are evaluated component by component.
#[derive(Clone, Copy, Debug, PartialEq)]
enum S {
    F(f32),
    I(i32),
    B(bool),
}

impl S {
    fn f(self) -> f32 {
        match self {
            S::F(x) => x,
            S::I(x) => x as f32,
            S::B(b) => b as i32 as f32,
        }
    }
    fn i(self) -> i32 {
        match self {
            S::I(x) => x,
            S::F(x) => x as i32,
            S::B(b) => b as i32,
        }
    }
    fn b(self) -> bool {
        match self {
            S::B(b) => b,
            S::F(x) => x != 0.0,
            S::I(x) => x != 0,
        }
    }
}

/// Components of a type (bool and unknown types: 1).
fn comps(ty: u8) -> usize {
    match ty {
        4..=7 => (ty - 3) as usize,
        8..=11 => (ty - 7) as usize,
        _ => 1,
    }
}

fn is_int(ty: u8) -> bool {
    matches!(ty, 8..=11)
}

/// Component c of a whole value (ints stay ints, bools bools; missing components read 0).
fn comp_of(v: &Value, c: usize) -> S {
    let c = if v.n() <= 1 { 0 } else { c };
    match *v {
        Value::F(n, a) => S::F(if c < n as usize { a[c] } else { 0.0 }),
        Value::I(n, a) => S::I(if c < n as usize { a[c] } else { 0 }),
        Value::B(b) => S::B(b),
        _ => S::F(0.0),
    }
}

fn typed(ty: u8, x: S) -> S {
    match ty {
        1 => S::B(x.b()),
        8..=11 => S::I(x.i()),
        4..=7 => S::F(x.f()),
        _ => x,
    }
}

/// Known components of some registers, shared by every pixel of a pixel processor.
#[derive(Clone, Default)]
pub struct Prefill(Vec<(usize, [Option<S>; 4])>);

/// Memo buffers an evaluator can reuse between evaluations (one per FX-Map render).
#[derive(Default)]
pub struct EvalBufs(Vec<[Option<S>; 4]>, Vec<Option<Value>>);

pub struct Eval<'a, E: Env> {
    f: &'a Func,
    /// Per register, per component.
    memo: Vec<[Option<S>; 4]>,
    /// Whole values of registers evaluated at once (samplers, loops, writes).
    whole: Vec<Option<Value>>,
    /// Variables: (index, value), few per function.
    pub vars: Vec<(u16, Value)>,
    /// The variables as a slot-indexed table, built at the first access (big pixel processors read
    /// and write thousands of slots per pixel; a scan of `vars` per access was the hot spot), and
    /// the slots in `vars` order (first writes appended) to rebuild `vars` at the end.
    vtab: Option<(Vec<Option<Value>>, Vec<u16>)>,
    /// variable accesses so far (the table is built from the 16th on: small functions keep the
    /// scan of `vars`), and whether the table was written (else `vars` stands)
    vacc: u32,
    vdirty: bool,

    env: &'a mut E,
    hoist_on: bool,
}

thread_local! {
    /// Memo buffers reused between evaluations (one function per pixel adds up).
    static SCRATCH: std::cell::RefCell<Vec<(Vec<[Option<S>; 4]>, Vec<Option<Value>>)>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl<'a, E: Env> Eval<'a, E> {
    pub fn new(f: &'a Func, env: &'a mut E) -> Self {
        let (memo, whole) = SCRATCH.with(|s| s.borrow_mut().pop()).unwrap_or_default();
        Self::with_bufs(f, env, memo, whole)
    }

    fn with_bufs(f: &'a Func, env: &'a mut E, mut memo: Vec<[Option<S>; 4]>, mut whole: Vec<Option<Value>>) -> Self {
        memo.clear();
        memo.resize(f.ins.len(), [None; 4]);
        whole.clear();
        whole.resize(f.ins.len(), None);
        // the untaken-branch rule holds for pixel processors too (scratch/pp2/sharedpp.py)
        let hoist_on = crate::env_once!("SDRT_NOHOIST").is_err();
        Eval { f, memo, whole, vars: Vec::new(), vtab: None, vacc: 0, vdirty: false, env, hoist_on }
    }

    pub fn run(f: &Func, env: &mut E) -> Value {
        Self::run_with_vars(f, env, &mut Vec::new())
    }

    /// As `run`, with variables that live on between evaluations: in an FX-Map, a function can
    /// set a variable that later functions of the same render read (counters, shared values).
    pub fn run_with_vars(f: &Func, env: &mut E, vars: &mut Vec<(u16, Value)>) -> Value {
        Self::run_vars_bufs(f, env, vars, None)
    }

    /// As `run_with_vars`, with the caller's memo buffers (no thread-local round trip).
    pub fn run_with_vars_in(f: &Func, env: &mut E, vars: &mut Vec<(u16, Value)>, bufs: &mut EvalBufs) -> Value {
        Self::run_vars_bufs(f, env, vars, Some(bufs))
    }

    fn run_vars_bufs(f: &Func, env: &mut E, vars: &mut Vec<(u16, Value)>, bufs: Option<&mut EvalBufs>) -> Value {
        if f.ins.is_empty() {
            return Value::Void;
        }
        let last = f.ins.len() - 1;
        let li = &f.ins[last];
        if li.op == 2 && li.ty == 0 {
            return Value::S(Func::literal(li, 0)); // a string input
        }
        let mut bufs = bufs;
        let mut e = match bufs.as_deref_mut() {
            Some(b) => Eval::with_bufs(f, env, std::mem::take(&mut b.0), std::mem::take(&mut b.1)),
            None => Eval::new(f, env),
        };
        e.vars = std::mem::take(vars);
        let v = e.value(last);
        if crate::env_once!("SDRT_REGDUMP").ok().and_then(|x| x.parse::<usize>().ok()) == Some(f.off) {
            static ONCE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if ONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2 {
                for (k, m) in e.memo.iter().enumerate() {
                    if m.iter().any(|x| x.is_some()) {
                        eprintln!("regdump r{} op{} {:?}", k, f.ins[k].op, m.iter().map(|x| x.map(|s| format!("{:?}", s))).collect::<Vec<_>>());
                    }
                }
            }
        }
        e.vtab_flush();
        *vars = std::mem::take(&mut e.vars);
        let memo = std::mem::take(&mut e.memo);
        let whole = std::mem::take(&mut e.whole);
        match bufs {
            Some(b) => {
                b.0 = memo;
                b.1 = whole;
            }
            None => SCRATCH.with(|s| s.borrow_mut().push((memo, whole))),
        }
        v
    }

    /// As `run_with_vars`, and returns the components of the registers marked in `keep` that this
    /// evaluation computed (for `run_prefilled`).
    pub fn run_capture(f: &Func, env: &mut E, vars: &mut Vec<(u16, Value)>, keep: &[bool]) -> (Value, Prefill) {
        Self::run_inner(f, env, vars, None, Some(keep))
    }

    /// As `run_with_vars`, with some registers' components known in advance.
    pub fn run_prefilled(f: &Func, env: &mut E, vars: &mut Vec<(u16, Value)>, pre: &Prefill) -> Value {
        Self::run_inner(f, env, vars, Some(pre), None).0
    }

    fn run_inner(f: &Func, env: &mut E, vars: &mut Vec<(u16, Value)>, pre: Option<&Prefill>, keep: Option<&[bool]>) -> (Value, Prefill) {
        if f.ins.is_empty() {
            return (Value::Void, Prefill::default());
        }
        let last = f.ins.len() - 1;
        let li = &f.ins[last];
        if li.op == 2 && li.ty == 0 {
            return (Value::S(Func::literal(li, 0)), Prefill::default());
        }
        let mut e = Eval::new(f, env);
        if let Some(p) = pre {
            for &(k, m) in &p.0 {
                e.memo[k] = m;
            }
        }
        e.vars = std::mem::take(vars);
        let v = e.value(last);
        e.vtab_flush();
        *vars = std::mem::take(&mut e.vars);
        let captured = match keep {
            Some(keep) => Prefill((0..f.ins.len()).filter(|&k| keep[k] && e.memo[k].iter().any(|x| x.is_some())).map(|k| (k, e.memo[k])).collect()),
            None => Prefill::default(),
        };
        let memo = std::mem::take(&mut e.memo);
        let whole = std::mem::take(&mut e.whole);
        SCRATCH.with(|s| s.borrow_mut().push((memo, whole)));
        (v, captured)
    }

    /// Variables are a flat array of scalar slots (NOTES: "var k as float4 covers slots k…k+3"): a
    /// variable of type ty at k reads slots k…k+n−1, unset slots read 0 (wood_base node 106 reads
    /// var 34 as a float2 after var 33 was set as a float3 and var 34 as a float).
    fn vtab(&mut self) -> &mut (Vec<Option<Value>>, Vec<u16>) {
        if self.vtab.is_none() {
            let mut t: Vec<Option<Value>> = Vec::new();
            let mut order = Vec::with_capacity(self.vars.len());
            for &(j, v) in &self.vars {
                if t.len() <= j as usize {
                    t.resize(j as usize + 1, None);
                }
                if t[j as usize].is_none() {
                    order.push(j);
                }
                t[j as usize] = Some(v);
            }
            self.vtab = Some((t, order));
        }
        self.vtab.as_mut().unwrap()
    }

    /// `vars` again from the table (after an evaluation that touched variables).
    fn vtab_flush(&mut self) {
        if let Some((t, order)) = self.vtab.take() {
            if self.vdirty {
                self.vars.clear();
                self.vars.extend(order.iter().map(|&j| (j, t[j as usize].unwrap())));
            }
        }
    }

    fn var_read(&mut self, k: u16, ty: u8) -> Option<Value> {
        let n = comps(ty).max(1);
        for c in 0..n as u16 {
            self.env.var_access(k + c, false);
        }
        self.vacc += 1;
        let scan = self.vtab.is_none() && self.vacc < 16;
        if !scan {
            self.vtab();
        }
        let tab: &[Option<Value>] = self.vtab.as_ref().map_or(&[], |t| &t.0[..]);
        let vars = &self.vars;
        let slot = |j: u16| if scan { vars.iter().rev().find(|(i, _)| *i == j).map(|(_, v)| *v) } else { tab.get(j as usize).copied().flatten() };
        if (0..n as u16).all(|c| slot(k + c).is_none()) {
            return None;
        }
        let sc = |c: usize| -> S {
            match slot(k + c as u16) {
                Some(Value::B(b)) => S::B(b),
                Some(Value::I(_, a)) => S::I(a[0]),
                Some(Value::F(_, a)) => S::F(a[0]),
                _ => S::F(0.0),
            }
        };
        Some(if ty == 1 {
            Value::B(sc(0).b())
        } else if is_int(ty) {
            let mut a = [0i32; 4];
            for c in 0..n {
                a[c] = sc(c).i();
            }
            Value::I(n as u8, a)
        } else {
            let mut a = [0.0f32; 4];
            for c in 0..n {
                a[c] = sc(c).f();
            }
            Value::F(n as u8, a)
        })
    }

    /// Set component c of variable k (a variable of type ty): slot k + c.
    fn set_var_comp(&mut self, k: u16, ty: u8, c: usize, x: S) {
        let n = comps(ty).max(1);
        let c = c.min(n - 1);
        let j = k + c as u16;
        self.env.var_access(j, true);
        let v = if ty == 1 { Value::B(x.b()) } else if is_int(ty) { Value::I(1, [x.i(), 0, 0, 0]) } else { Value::F(1, [x.f(), 0.0, 0.0, 0.0]) };
        if crate::env_once!("SDRT_VARLOG").ok().and_then(|v| v.parse::<u16>().ok()) == Some(j) {
            eprintln!("varlog set slot {} = {:?} by fn {}", j, v, self.f.off);
        }
        self.vacc += 1;
        if self.vtab.is_none() && self.vacc < 16 {
            match self.vars.iter().position(|(i, _)| *i == j) {
                Some(p) => self.vars[p].1 = v,
                None => self.vars.push((j, v)),
            }
            return;
        }
        self.vdirty = true;
        let (t, order) = self.vtab();
        if t.len() <= j as usize {
            t.resize(j as usize + 1, None);
        }
        if t[j as usize].is_none() {
            order.push(j);
        }
        t[j as usize] = Some(v);
    }

    /// The whole value of register k: its components in order.
    fn value(&mut self, k: usize) -> Value {
        if k >= self.f.ins.len() {
            return Value::Void;
        }
        let ty = self.f.ins[k].ty;
        let n = comps(ty);
        let mut cs = [S::F(0.0); 4];
        for c in 0..n {
            cs[c] = self.comp(k, c);
        }
        match ty {
            1 => Value::B(cs[0].b()),
            8..=11 => Value::I(n as u8, std::array::from_fn(|c| if c < n { cs[c].i() } else { 0 })),
            _ => Value::F(n as u8, std::array::from_fn(|c| if c < n { cs[c].f() } else { 0.0 })),
        }
    }

    /// Component c of register k, computed on demand (an operand's component is only evaluated
    /// when a result component needs it — measured with sbsrender: lab/py/lazyprobe.py,
    /// orderprobe.py).
    fn comp(&mut self, k: usize, c: usize) -> S {
        if k >= self.f.ins.len() {
            return S::F(0.0); // 0xFFFF: no operand (e.g. the second of a sequence)
        }
        let n = comps(self.f.ins[k].ty);
        let c = if n == 1 { 0 } else { c.min(3) };
        if let Some(v) = self.memo[k][c] {
            return v;
        }
        let v = self.compute(k, c);
        self.memo[k][c] = Some(v);
        v
    }

    /// The whole value of register k, computed once (for ops that need every component).
    fn whole_of(&mut self, k: usize) -> Value {
        if k >= self.f.ins.len() {
            return Value::Void;
        }
        if let Some(v) = self.whole[k] {
            return v;
        }
        let v = self.value(k);
        self.whole[k] = Some(v);
        v
    }

    /// Forget the registers a loop's condition and body depend on, so they are computed again.
    /// Forget the registers k depends on that can change between iterations (volatile registers,
    /// see `Func::decode`; the others would be recomputed to the same values without side effects; lists built at
    /// decode, `Func::loop_inv`).
    fn invalidate(&mut self, k: usize) {
        let f = self.f;
        if let Some((_, list)) = f.loop_inv.iter().find(|e| e.0 == k) {
            for &r in list {
                self.memo[r] = [None; 4];
                self.whole[r] = None;
            }
        } else if k < f.ins.len() {
            self.memo[k] = [None; 4];
            self.whole[k] = None;
        }
    }

    fn compute(&mut self, k: usize, c: usize) -> S {
        let i = &self.f.ins[k];
        let ty = i.ty;
        let op = |j: usize| i.ops[j] as usize;
        match i.op {
            // bool: one u16; without it the constant is true (scratch/pp2/boolc.py; multiwood node 24)
            0 if ty == 1 => S::B(i.ops.first().map_or(true, |&x| x != 0)),
            0 => {
                // constant: value words after the pad
                let w = self.f.lits[k][c.min(comps(ty) - 1)];
                if is_int(ty) { S::I(w as i32) } else { S::F(f32::from_bits(w)) }
            }
            1 => typed(ty, comp_of(&self.env.sysvar(i.ops[0]), c)),
            2 => {
                let uid = self.f.lits[k][0];
                typed(ty, comp_of(&self.env.input(uid), c))
            }
            3 => typed(ty, comp_of(&self.env.vp_value(i.ops[0] as usize), c)), // u16 node index
            4 => {
                // a variable is read whole at the first demand of any component: a later setvar of
                // the same variable does not change this register (matches sbsrender on worn_rope)
                if self.whole[k].is_none() {
                    let v = self.var_read(i.ops[0], ty).unwrap_or(Value::F(4, [0.0; 4]));
                    if crate::env_once!("SDRT_VARLOG").ok().and_then(|x| x.parse::<u16>().ok()).is_some_and(|j| j >= i.ops[0] && j < i.ops[0] + comps(ty) as u16) {
                        eprintln!("varlog read var {} -> {:?} by fn {}", i.ops[0], v, self.f.off);
                    }
                    self.whole[k] = Some(v);
                }
                typed(ty, comp_of(&self.whole[k].unwrap(), c))
            }
            6 => {
                let v = self.whole_of(op(0));
                if self.whole[k].is_none() {
                    self.env.write(i.ops.get(1).copied().unwrap_or(0), v);
                    self.whole[k] = Some(v);
                }
                comp_of(&v, c)
            }
            7 => {
                let x = self.comp(op(0), c);
                self.set_var_comp(i.ops[1], ty, c, x);
                x
            }
            9 => {
                let cond = self.comp(op(0), 0).b();
                // Visit branches in their encoded order: true, then false. The chosen branch runs
                // normally; in the other branch only shared expressions are materialized.
                // Single-use expressions stay lazy. Always visiting the chosen branch first
                // consumes the right number of draws but can assign them to the wrong values
                // (multiwood FX-Map 64, size.y). scratch/2026-10-29/branch_order.py.
                let taken = if cond { Some(self.comp(op(1), c)) } else { None };
                if self.hoist_on {
                    // the untaken branch's component c, scalar by scalar: its shared components
                    let ins = &self.f.ins;
                    let u = if cond { op(2) } else { op(1) };
                    let mut stack = if u < ins.len() { vec![(u, clamp_comp(ins, u, c))] } else { vec![] };
                    let mut found = vec![];
                    let mut reads = Vec::new();
                    while let Some((r, rc)) = stack.pop() {
                        if found.contains(&(r, rc)) {
                            continue;
                        }
                        if self.f.hoist[r][rc] {
                            // a shared setvar / write / loop there does not run (wood_base node 106)
                            if !matches!(ins[r].op, 6 | 7 | 11) {
                                found.push((r, rc));
                            }
                            continue;
                        }
                        reads.clear();
                        scalar_reads(ins, &self.f.side, r, rc, &mut reads);
                        stack.extend(reads.iter().rev());
                    }
                    for (r, rc) in found {
                        self.comp(r, rc);
                    }
                }
                taken.unwrap_or_else(|| self.comp(op(2), c))
            }
            11 => {
                // loop: init, condition, body, max iterations (0: no limit); run once, whole values
                if self.whole[k].is_none() {
                    let max = Func::literal(&Ins { op: 0, ty: 0, ops: Ops::from(&i.ops[3..]) }, 0);
                    let (init, cond, body) = (op(0), op(1), op(2));
                    let mut last = self.whole_of(init);
                    let mut it = 0u32;
                    // a do-while: the body runs once before the condition is tested (stylized_sand_01_pkg
                    // PPs 49/52, v9: the condition is false from the start, yet the reference ran the
                    // body; 1/512 → exact). SDRT_LOOPDO=0: a while loop
                    // the condition ENDS the loop: while (!cond) (kachelboden 360/378, v9: the condition
                    // is true from the start and the reference never runs the body — its output stays
                    // 0; stylized_sand_01_pkg 49/52, false from the start: the body runs, which a
                    // do-while also explained). SDRT_LOOPDO=do: do-while (cond), =0: while (cond)
                    let mode = crate::env_once!("SDRT_LOOPDO").ok();
                    let do_while = mode == Some("do");
                    let until = mode.is_none() || mode == Some("until");
                    loop {
                        self.invalidate(cond);
                        self.invalidate(body);
                        if until {
                            if self.comp(cond, 0).b() {
                                break;
                            }
                        } else if !(do_while && it == 0) && !self.comp(cond, 0).b() {
                            break;
                        }
                        last = self.value(body);
                        it += 1;
                        if (max != 0 && it >= max) || it >= 1 << 24 {
                            break;
                        }
                    }
                    self.whole[k] = Some(last);
                }
                let v = self.whole[k].unwrap();
                comp_of(&v, c)
            }
            12 => {
                // sequence: each operand that has side effects runs once, whole — the second too, even
                // when the sequence's own type has fewer components (scratch/pp2/setvar.py case C)
                let (a, b) = (op(0), op(1));
                if self.f.side.get(a).copied().unwrap_or(false) && self.whole[a].is_none() {
                    self.whole_of(a);
                }
                if self.f.side.get(b).copied().unwrap_or(false) && self.whole[b].is_none() {
                    self.whole_of(b);
                }
                self.comp(b, c)
            }
            13 => {
                let (a, b) = (op(0), op(1));
                let na = if a < self.f.ins.len() { comps(self.f.ins[a].ty) } else { 0 };
                let x = if c < na { self.comp(a, c) } else { self.comp(b, c - na) };
                typed(ty, x)
            }
            15 => typed(ty, self.comp(op(c.min(3)), 0)),
            16 => {
                let sel = i.ops[1];
                self.comp(op(0), ((sel >> (2 * c)) & 3) as usize)
            }
            17 => {
                // toint floors (−1.5 → −2; lab/py/semprobe.py)
                let x = self.comp(op(0), c);
                match (ty, x) {
                    (8..=11, S::F(v)) => S::I(v.floor() as i32),
                    _ => typed(ty, x),
                }
            }
            18..=22 | 48 | 49 | 54 => {
                // a scalar operand of a vector op is extended with zeros, except for mul (also
                // mulscalar), which broadcasts it (scratch/pp2/bcast3.py: float2 + float = (x + s, y))
                let zero_ext = |vm: &Self, r: usize| -> bool {
                    i.op != 20 && c > 0 && r < vm.f.ins.len() && comps(vm.f.ins[r].ty) == 1 && comps(ty) > 1
                };
                let zero = |vm: &Self, r: usize| -> S { if is_int(vm.f.ins[r].ty) { S::I(0) } else { S::F(0.0) } };
                let a = if zero_ext(self, op(0)) { zero(self, op(0)) } else { self.comp(op(0), c) };
                let b = if zero_ext(self, op(1)) { zero(self, op(1)) } else { self.comp(op(1), c) };
                // float mod by 0: x on the GPU (pixel processors), NaN in the CPU evaluator
                if i.op == 22 && !self.env.cpu_trig() && !is_int(ty) && b.f() == 0.0 {
                    return a;
                }
                scalar_binary(i.op, ty, a, b)
            }
            24 => {
                // dot: component-wise products in order
                let (a, b) = (op(0), op(1));
                let n = comps(self.f.ins[a].ty).min(comps(self.f.ins[b].ty)).max(1);
                let ints = is_int(self.f.ins[a].ty) && is_int(self.f.ins[b].ty);
                let (mut fs, mut is) = (0.0f32, 0i32);
                for j in 0..n {
                    let (x, y) = (self.comp(a, j), self.comp(b, j));
                    fs += x.f() * y.f();
                    is = is.wrapping_add(x.i().wrapping_mul(y.i()));
                }
                if ints && is_int(ty) { S::I(is) } else { typed(ty, S::F(fs)) }
            }
            23 => match self.comp(op(0), c) {
                S::I(x) => S::I(-x),
                x => S::F(-x.f()),
            },
            // and / or evaluate both operands, first then second (no short circuit: a rand in the
            // second always draws; lab/py/shortprobe.py)
            26 => {
                let (a, b) = (self.comp(op(0), 0).b(), self.comp(op(1), 0).b());
                S::B(a && b)
            }
            27 => {
                let (a, b) = (self.comp(op(0), 0).b(), self.comp(op(1), 0).b());
                S::B(a || b)
            }
            28 => S::B(!self.comp(op(0), 0).b()),
            29..=34 => {
                let a = self.comp(op(0), 0);
                let b = self.comp(op(1), 0);
                let (x, y) = match (a, b) {
                    (S::I(p), S::I(q)) => (p as f64, q as f64),
                    _ => (a.f() as f64, b.f() as f64),
                };
                S::B(match i.op {
                    29 => x == y,
                    30 => x != y,
                    31 => x > y,
                    32 => x >= y,
                    33 => x < y,
                    _ => x <= y,
                })
            }
            35..=44 | 53 => {
                let a = self.comp(op(0), c);
                if let (S::I(x), 35) = (a, i.op) {
                    return S::I(x.abs());
                }
                let cpu = self.env.cpu_trig();
                // Domain guards measured across 14.1 vk and 14.1 sse2 (scratch/t/cpumath*, vpmath):
                // sqrt(x <= 0) = 0 on both CPU and GPU; ln(x <= 0) = 0 and log2(x <= 0) = 0 in the
                // CPU evaluator (NodeEnv / QuadEnv, e.g. test-2-twm 961 size fn log2(0) -> 0).
                let f: fn(f32) -> f32 = match i.op {
                    35 => f32::abs,
                    36 => f32::floor,
                    37 => f32::ceil,
                    38 if cpu => sbs_cos,
                    39 if cpu => sbs_sin,
                    38 => f32::cos,
                    39 => f32::sin,
                    40 => |x| if x <= 0.0 { 0.0 } else { x.sqrt() },
                    41 if cpu => |x| if x <= 0.0 { 0.0 } else { x.ln() },
                    41 => f32::ln,
                    42 => f32::exp,
                    43 if crate::env_once!("SDRT_OP43").as_deref() == Ok("sq") => |x: f32| x * x,
                    43 => f32::exp2,
                    44 => f32::tan,
                    _ if cpu => |x| if x <= 0.0 { 0.0 } else { x.log2() },
                    _ => f32::log2,
                };
                typed(ty, S::F(f(a.f())))
            }
            45 => {
                // atan2(v.y, v.x), measured (lab/py/semprobe.py)
                let x = self.comp(op(0), 0).f();
                let y = self.comp(op(0), 1).f();
                S::F(y.atan2(x))
            }
            46 => {
                // cartesian: (rho, theta) → (rho cos θ, rho sin θ)
                let rho = self.comp(op(0), 0).f();
                let th = self.comp(op(1), 0).f();
                let (cs, sn) = if self.env.cpu_trig() { (sbs_cos(th), sbs_sin(th)) } else { (th.cos(), th.sin()) };
                S::F(if c == 0 { rho * cs } else { rho * sn })
            }
            47 => {
                let a = self.comp(op(0), c).f();
                let b = self.comp(op(1), c).f();
                let t = self.comp(op(2), c).f();
                S::F(a + (b - a) * t)
            }
            50 => {
                let a = self.comp(op(0), c).f();
                let r = self.env.rand();
                if crate::env_once!("SDRT_RANDLOG").is_ok() {
                    eprintln!("randlog r{} c{} {}", k, c, r);
                }
                S::F(a * r)
            }
            51 | 52 => {
                if self.whole[k].is_none() {
                    let px = self.comp(op(0), 0).f();
                    let py = self.comp(op(0), 1).f();
                    let input = i.ops[1] as usize;
                    // no third operand: nearest (texel floor(pos·size)); 1: bilinear
                    // (scratch/pp2/samp2.py, 2026-10-28: the reverse of what was assumed before)
                    // the filter operand is a bit field: bit 0 bilinear (else nearest), bit 1 edges
                    // clamped (else tiled) — foliage 2039/2048 (v9, operand 3): mean 0.064 → exact.
                    // Only 0 and 1 had been seen before. SDRT_SAMPF=old: bilinear only for 1, tiled
                    let old = crate::env_once!("SDRT_SAMPF").as_deref() == Ok("old");
                    let fop = i.ops.get(2).copied().unwrap_or(0);
                    let nearest = if old { fop != 1 } else { fop & 1 == 0 };
                    self.env.sample_filter(if old { fop & 1 } else { fop });
                    let s = self.env.sample(input, [px, py], i.op == 52, nearest);
                    self.whole[k] = Some(if i.op == 51 { Value::f1(s[0]) } else { Value::F(4, s) });
                }
                let v = self.whole[k].unwrap();
                comp_of(&v, c)
            }
            _ => S::F(0.0),
        }
    }
}

fn scalar_binary(op: u8, ty: u8, a: S, b: S) -> S {
    let ints = is_int(ty) || (matches!(a, S::I(_)) && matches!(b, S::I(_)));
    if ints {
        let (p, q) = (a.i(), b.i());
        return S::I(match op {
            18 => p.wrapping_add(q),
            19 => p.wrapping_sub(q),
            20 => p.wrapping_mul(q),
            21 => if q != 0 { p.wrapping_div(q) } else { 0 },
            // int division truncates, int mod is floored (−7 mod 2 = 1; semprobe.py)
            22 => if q != 0 { let r = p.wrapping_rem(q); if r != 0 && (r < 0) != (q < 0) { r + q } else { r } } else { 0 },
            48 => p.min(q),
            49 => p.max(q),
            54 => (p as f32).powf(q as f32) as i32,
            _ => 0,
        });
    }
    let (p, q) = (a.f(), b.f());
    S::F(match op {
        18 => p + q,
        19 => p - q,
        20 => p * q,
        21 => p / q,
        // GLSL mod (x − y·floor(x/y)), measured
        22 => match crate::env_once!("SDRT_FMOD").as_deref().unwrap_or("") {
            "fmod" => { let r = p % q; if r != 0.0 && (r < 0.0) != (q < 0.0) { r + q } else { r } }
            "trunc" => p % q,
            "fma" => (-q).mul_add((p / q).floor(), p),
            "f64" => ((p as f64) - (q as f64) * ((p as f64) / (q as f64)).floor()) as f32,
            "f64q" => ((p as f64) - (q as f64) * ((p / q) as f64).floor()) as f32,
            _ => p - q * (p / q).floor(),
        },
        48 => p.min(q),
        49 => p.max(q),
        // pow(0, q) = 0 (incl. pow(0, 0) = 0, pow(0, -1) = 0) and pow(p < 0, non-int) = 0 on both CPU & GPU
        54 => if !p.is_nan() && !q.is_nan() && (p == 0.0 || (p < 0.0 && q != q.floor())) { 0.0 } else { p.powf(q) },
        _ => 0.0,
    })
}

#[allow(dead_code)]
fn binary(op: u8, ty: u8, a: Value, b: Value) -> Value {
    let n = a.n().max(b.n()).max(1);
    let ints = matches!(ty, 8..=11) || (matches!(a, Value::I(..)) && matches!(b, Value::I(..)) && op != 24);
    if ints {
        let (x, y) = (a.iv(), b.iv());
        let (sa, sb) = (a.n() == 1, b.n() == 1);
        let mut v = [0i32; 4];
        for c in 0..n {
            let p = if sa { x[0] } else { x[c] };
            let q = if sb { y[0] } else { y[c] };
            v[c] = match op {
                18 => p.wrapping_add(q),
                19 => p.wrapping_sub(q),
                20 => p.wrapping_mul(q),
                21 => if q != 0 { p.wrapping_div(q) } else { 0 },
                22 => if q != 0 { p.wrapping_rem(q) } else { 0 },
                48 => p.min(q),
                49 => p.max(q),
                54 => (p as f32).powf(q as f32) as i32,
                _ => 0,
            };
        }
        if op == 24 {
            return Value::I(1, [(0..n).map(|c| x[c] * y[c]).sum(), 0, 0, 0]);
        }
        return Value::I(n as u8, v);
    }
    let (x, y) = (a.fv(), b.fv());
    if op == 24 {
        let m = a.n().min(b.n()).max(1);
        return Value::f1((0..m).map(|c| x[c] * y[c]).sum());
    }
    let (sa, sb) = (a.n() == 1, b.n() == 1);
    let mut v = [0.0f32; 4];
    for c in 0..n {
        let p = if sa { x[0] } else { x[c] };
        let q = if sb { y[0] } else { y[c] };
        v[c] = match op {
            18 => p + q,
            19 => p - q,
            20 => p * q,
            21 => p / q,
            // TODO measure: fmod or GLSL mod
            22 => p - q * (p / q).floor(),
            48 => p.min(q),
            49 => p.max(q),
            54 => p.powf(q),
            _ => 0.0,
        };
    }
    Value::F(n as u8, v)
}
