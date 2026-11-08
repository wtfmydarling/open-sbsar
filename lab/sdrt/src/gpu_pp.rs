//! Scalar-demand PP code generation. The traversal order mirrors vm::Eval::comp, including
//! component memoization and rand draws. Unsupported control flow stays on the CPU as a whole.
//!
//! Two forms. The static form (no conditional side effects) is straight-line code: one `let`
//! per register component, `select` for conditionals. The control-flow form (conditional rand,
//! variable mutation, loops) keeps each register component in a function-scope variable with a
//! validity flag and emits its computation in one place: the generator tracks statically which
//! components are computed at each point (`St`), so a component needs a runtime check only
//! where that depends on the path taken. A conditional's two operands are emitted one after
//! the other, each register under the predicate for which the VM computes it (taken branch, or
//! the untaken branch's shared components), so nothing is duplicated per branch.
//!
//! Per-node values (function literals, graph input and value-processor values, the variable
//! state after the size/seed functions) are read from `config.lits`, so functions of the same
//! structure share one pipeline. A value stays in the WGSL text when the driver's compiler could
//! compute differently with it known (`Konst`, `Use`): every result stays as it was with all
//! values embedded.
use crate::vm::{Func, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    F,
    I,
    B,
}

/// What the driver's compiler can know about a value: computed from per-node values only
/// (`deps`, indices into `Compiler::lits`) and shader constants, so it may fold it.
#[derive(Clone, PartialEq, Debug)]
struct Konst {
    deps: Vec<usize>,
    /// computing it at run time from the values gives the folded result bit for bit: no float
    /// arithmetic (the driver folds known float expressions with other roundings), only
    /// conversions, comparisons, min/max, floor/ceil/abs, integer operations
    safe: bool,
}
type K = Option<Rc<Konst>>;
fn kconst() -> K {
    Some(Rc::new(Konst { deps: vec![], safe: true }))
}
/// An operation on its operands: known when all of them are.
fn kjoin(ops: &[&K], safe: bool) -> K {
    let mut deps = Vec::new();
    let mut all = safe;
    for k in ops {
        let k = (*k).as_ref()?;
        deps.extend_from_slice(&k.deps);
        all &= k.safe;
    }
    deps.sort_unstable();
    deps.dedup();
    Some(Rc::new(Konst { deps, safe: all }))
}
/// A value that is one of several (a branch result, a variable): it may be known when any is.
fn kany(ops: &[&K]) -> K {
    let known: Vec<&Rc<Konst>> = ops.iter().filter_map(|k| k.as_ref()).collect();
    match known[..] {
        [] => None,
        [one] => Some(one.clone()),
        _ => {
            let mut deps: Vec<usize> = known.iter().flat_map(|k| k.deps.iter().copied()).collect();
            deps.sort_unstable();
            deps.dedup();
            Some(Rc::new(Konst { deps, safe: known.iter().all(|k| k.safe) }))
        }
    }
}
/// How an operation uses a known operand when its result is not known: whether the operand's
/// values must stay in the text (the driver specializes on them) or may be read at run time.
#[derive(Clone, Copy)]
enum Use {
    /// a use that gives the same result with the value known or read: embedded only when the
    /// operand itself is not `safe` (arithmetic operands are handled by `arith`)
    Plain,
    /// pow and division operands, conditions, loop values: always embedded
    Exact,
}

#[derive(Clone)]
struct Expr {
    text: String,
    kind: Kind,
    konst: K,
}
fn rt(text: String, kind: Kind) -> Expr {
    Expr { text, kind, konst: None }
}
fn kind(ty: u8) -> Kind {
    match ty {
        1 => Kind::B,
        8..=11 => Kind::I,
        _ => Kind::F,
    }
}
fn typename(k:Kind)->&'static str {match k {Kind::F=>"f32",Kind::I=>"i32",Kind::B=>"bool"}}
fn pack(e:&Expr)->String {if e.kind==Kind::B {format!("select(0u,1u,{})",e.text)} else {format!("bitcast<u32>({})",e.text)}}
fn unpack(text:String,k:Kind,konst:K)->Expr {Expr {text:if k==Kind::B {format!("({text}!=0u)")} else {format!("bitcast<{}>({text})",typename(k))},kind:k,konst}}
fn comps(ty: u8) -> usize {
    match ty {
        4..=7 => (ty - 3) as usize,
        8..=11 => (ty - 7) as usize,
        _ => 1,
    }
}
fn lit(x: f32) -> Expr {
    Expr { text: format!("bitcast<f32>(0x{:08x}u)", x.to_bits()), kind: Kind::F, konst: kconst() }
}
fn cast(x: &Expr, k: Kind) -> String {
    if x.kind == k {
        return x.text.clone();
    }
    match (x.kind, k) {
        (Kind::B, Kind::F) => format!("select(0.0,1.0,{})", x.text),
        (Kind::B, Kind::I) => format!("select(0i,1i,{})", x.text),
        (Kind::F, Kind::I) => format!("to_int({})", x.text),
        (Kind::I, Kind::F) => format!("f32({})", x.text),
        (_, Kind::B) => format!("({} != {})", x.text, if x.kind == Kind::I { "0i" } else { "0.0" }),
        _ => unreachable!(),
    }
}
/// `cast` with the operand's compile-time knowledge (conversions are exact).
fn castk(x: &Expr, k: Kind) -> Expr {
    Expr { text: cast(x, k), kind: k, konst: x.konst.clone() }
}
/// Component c of a value: its bits and kind (no number: a float zero).
fn component(v: Value, c: usize) -> (u32, Kind) {
    let c = if v.n() <= 1 { 0 } else { c };
    match v {
        Value::I(n, a) => (if c < n as usize { a[c] as u32 } else { 0 }, Kind::I),
        Value::B(b) => (u32::from(b), Kind::B),
        Value::F(n, a) => ((if c < n as usize { a[c] } else { 0. }).to_bits(), Kind::F),
        _ => (0, Kind::F),
    }
}
/// Marks a per-node value in the generated text; resolved by `Compiler::resolve`.
const TOKEN: char = '\u{1}';

thread_local! {
    /// The graph node being compiled (research switch SDRT_GPU_PIN=N:...; set by ops.rs).
    pub static NODE: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

/// What the control-flow form knows about a register component (or whole value) at a point of
/// the generated code. `Cond(p)`: computed at least when predicate p holds.
#[derive(Clone, Copy, PartialEq, Debug)]
enum St {
    No,
    Yes,
    Cond(usize),
    Maybe,
}
struct Pred {
    name: String,
    parent: Option<usize>,
}
/// The generator state a loop body is regenerated from (`Compiler::save`).
#[derive(Clone)]
struct Saved {
    st: Vec<[St; 4]>,
    wst: Vec<St>,
    lits: usize,
    pinned: HashSet<usize>,
    mk: HashMap<(usize, usize), K>,
    mkind: HashMap<(usize, usize), Kind>,
    wk: HashMap<usize, [K; 4]>,
    slot_next: HashMap<u16, Slot>,
    chains: HashMap<String, Rc<Vec<usize>>>,
    wchain: HashMap<usize, Vec<usize>>,
    serial: usize,
    preds: usize,
    negs: HashMap<usize, usize>,
}

struct Compiler<'a> {
    f: &'a Func,
    input: &'a dyn Fn(u32) -> Value,
    vp: &'a dyn Fn(usize) -> Value,
    vars: HashMap<u16, Expr>,
    /// variable state after the size/seed functions (static form: read on first use)
    initial: HashMap<u16, (u32, Kind)>,
    /// Per-node values in generation order: (bits, kind; None: raw u32 bits). The text refers to
    /// them by token; `pinned` ones are embedded as before, the others read from `config.lits`.
    lits: Vec<(u32, Option<Kind>)>,
    pinned: HashSet<usize>,
    reads: HashMap<usize, Vec<Expr>>,
    memo: HashMap<(usize, usize), Expr>,
    samples: HashMap<usize, String>,
    side: Vec<bool>,
    body: String,
    depth: usize,
    dynamic: bool,
    serial: usize,
    var_ids: HashMap<u16,usize>,
    // control-flow form
    volatile: Vec<bool>,
    nohoist: bool,
    st: Vec<[St; 4]>,
    wst: Vec<St>,
    preds: Vec<Pred>,
    negs: HashMap<usize, usize>,
    /// predicates known to hold where code is being emitted (with their ancestors)
    known: Vec<usize>,
    decls: BTreeMap<String, &'static str>,
    mk: HashMap<(usize, usize), K>,
    mkind: HashMap<(usize, usize), Kind>,
    wk: HashMap<usize, [K; 4]>,
    /// per variable slot, what the driver may know of its value and the values that entered
    /// it through arithmetic: assumed (`slot_k`, from the previous pass) and found by this
    /// pass's stores (`slot_next`)
    slot_k: HashMap<u16, Slot>,
    slot_next: HashMap<u16, Slot>,
    /// per runtime value (by its WGSL name), the per-node values that entered it through float
    /// arithmetic (`arith`); per whole register value the same; the result of the last compute
    chains: HashMap<String, Rc<Vec<usize>>>,
    wchain: HashMap<usize, Vec<usize>>,
    chain_out: Vec<usize>,
}
type Slot = (K, Vec<usize>);
fn merge_ids(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut v: Vec<usize> = a.iter().chain(b).copied().collect();
    v.sort_unstable();
    v.dedup();
    v
}
impl Compiler<'_> {
    /// A per-node value: a token for the next `lits` entry, read as kind k.
    fn value(&mut self, bits: u32, k: Kind) -> Expr {
        let n = self.lits.len();
        self.lits.push((bits, Some(k)));
        Expr { text: format!("{TOKEN}{n}{TOKEN}"), kind: k, konst: Some(Rc::new(Konst { deps: vec![n], safe: true })) }
    }
    fn component_value(&mut self, v: Value, c: usize) -> Expr {
        let (bits, k) = component(v, c);
        self.value(bits, k)
    }
    /// The values a known operand depends on stay in the text when the operation uses it in a
    /// way the driver could compute differently knowing them (see `Use`).
    fn pin_use(&mut self, e: &Expr, u: Use) {
        let Some(q) = &e.konst else { return };
        let pin = !q.safe
            || match u {
                Use::Plain => false,
                Use::Exact => true,
            };
        if pin {
            self.pinned.extend(q.deps.iter().copied());
        }
    }
    /// The result's knowledge; when it is not known, its known operands meet their uses here.
    fn finish(&mut self, result: K, ops: &[(&Expr, Use)]) -> K {
        if result.is_none() {
            for (e, u) in ops {
                self.pin_use(e, *u);
            }
        }
        result
    }
    /// The per-node values that entered a value: its own when known, else those recorded
    /// for its name by `arith`.
    fn entries(&self, e: &Expr) -> Vec<usize> {
        match &e.konst {
            Some(q) => q.deps.clone(),
            None => self.chains.get(&e.text).map_or_else(Vec::new, |c| (**c).clone()),
        }
    }
    fn entries_of(&self, ops: &[&Expr]) -> Vec<usize> {
        ops.iter().fold(Vec::new(), |acc, e| merge_ids(&acc, &self.entries(e)))
    }
    /// A float add/sub/mul/div/mod (or dot, lerp, rand scale) with an unknown result. The driver
    /// reassociates and distributes a tree of them, combining the values it knows (stripe-new-1
    /// node 341: 0.5 - (x - 0.5) * 2 in the last bit), and may fold identities with 0 and 1: the
    /// values of a tree that holds two or more of them, and 0, 1 and non-finite values, stay
    /// embedded. Returns the values that entered the result.
    fn arith(&mut self, ops: &[&Expr]) -> Vec<usize> {
        let mut all = Vec::new();
        let mut n = 0;
        for e in ops {
            let x = self.entries(e);
            if e.konst.is_some() {
                for &id in &x {
                    let v = f32::from_bits(self.lits[id].0);
                    if !v.is_finite() || v == 0.0 || v.abs() == 1.0 {
                        self.pinned.insert(id);
                    }
                }
            }
            if !x.is_empty() {
                n += 1;
                all = merge_ids(&all, &x);
            }
        }
        if n >= 2 {
            self.pinned.extend(all.iter().copied());
        }
        all
    }
    /// select(b, a, cond): known operands stay embedded. A known condition is folded (and the
    /// driver may contract across the select), and a known value changes how the select is
    /// lowered and what the other value's instructions fuse with (abstract_01 node 158:
    /// select(x, -1, c) differs in the last bit with -1 read at run time).
    fn select_konst(&mut self, cond: &Expr, a: &Expr, b: &Expr) -> K {
        for e in [cond, a, b] {
            self.pin_use(e, Use::Exact);
        }
        kjoin(&[&cond.konst, &a.konst, &b.konst], true)
    }
    fn line(&mut self, text: &str) {
        self.body.push_str("    ");
        self.body.push_str(text);
        self.body.push('\n');
    }
    fn clamp(&self, k: usize, c: usize) -> usize {
        if comps(self.f.ins[k].ty) == 1 { 0 } else { c.min(3) }
    }
    /// Variable slot `slot` as read by getvar (control-flow form).
    fn slot_konst(&self, slot: u16) -> Slot {
        self.slot_k.get(&slot).cloned().unwrap_or_else(|| (kconst(), vec![]))
    }

    // ---- control-flow form: static computed-state tracking ----
    fn yes(&self, s: St) -> bool {
        match s {
            St::Yes => true,
            St::Cond(q) => self.known.contains(&q),
            _ => false,
        }
    }
    fn memo_expr(&self, k: usize, c: usize) -> Expr {
        Expr { text: format!("m{k}_{c}"), kind: self.mkind.get(&(k, c)).copied().unwrap_or(Kind::F), konst: self.mk.get(&(k, c)).cloned().flatten() }
    }
    fn whole_comp(&mut self, k: usize, c: usize, kd: Kind) -> Expr {
        let konst = self.wk.get(&k).and_then(|w| w[c.min(3)].clone());
        let e = unpack(format!("w{k}[{c}u]"), kd, konst);
        if let Some(ch) = self.wchain.get(&k).filter(|ch| !ch.is_empty()) {
            self.chains.insert(e.text.clone(), Rc::new(ch.clone()));
        }
        e
    }
    fn pred(&mut self, parent: Option<usize>, cond: &str) -> usize {
        let id = self.preds.len();
        let name = format!("p{id}");
        let value = match parent {
            Some(q) => format!("{} && {cond}", self.preds[q].name),
            None => cond.to_string(),
        };
        self.line(&format!("let {name}={value};"));
        self.preds.push(Pred { name, parent });
        id
    }
    fn neg(&mut self, p: usize) -> usize {
        if let Some(&n) = self.negs.get(&p) {
            return n;
        }
        let id = self.preds.len();
        self.preds.push(Pred { name: format!("(!{})", self.preds[p].name), parent: None });
        self.negs.insert(p, id);
        self.negs.insert(id, p);
        id
    }
    fn push_known(&mut self, p: usize) -> usize {
        let mut n = 0;
        let mut q = Some(p);
        while let Some(x) = q {
            self.known.push(x);
            n += 1;
            q = self.preds[x].parent;
        }
        n
    }
    fn states(&self) -> (Vec<[St; 4]>, Vec<St>) {
        (self.st.clone(), self.wst.clone())
    }
    /// `if p { f }`: afterwards, what f computed is known to be computed when p holds.
    fn under<T>(&mut self, p: usize, f: impl FnOnce(&mut Self) -> Result<T, String>) -> Result<T, String> {
        let (st0, wst0) = self.states();
        let head = format!("if {} {{", self.preds[p].name);
        self.line(&head);
        let n = self.push_known(p);
        let r = f(self)?;
        let merge = |s: &Self, before: St, inside: St| {
            if before == inside {
                before
            } else if s.yes(inside) {
                if before == St::Yes { St::Yes } else { St::Cond(p) }
            } else {
                St::Maybe
            }
        };
        let st: Vec<[St; 4]> = (0..self.st.len()).map(|k| std::array::from_fn(|c| merge(self, st0[k][c], self.st[k][c]))).collect();
        let wst: Vec<St> = (0..self.wst.len()).map(|k| merge(self, wst0[k], self.wst[k])).collect();
        self.known.truncate(self.known.len() - n);
        self.line("}");
        self.st = st;
        self.wst = wst;
        Ok(r)
    }
    /// `if p { a } else { b }`.
    fn under_else(&mut self, p: usize, a: impl FnOnce(&mut Self) -> Result<(), String>, b: impl FnOnce(&mut Self) -> Result<(), String>) -> Result<(), String> {
        let (st0, wst0) = self.states();
        let np = self.neg(p);
        let head = format!("if {} {{", self.preds[p].name);
        self.line(&head);
        let n = self.push_known(p);
        a(self)?;
        let ya: Vec<[bool; 4]> = self.st.iter().map(|s| s.map(|x| self.yes(x))).collect();
        let wya: Vec<bool> = self.wst.iter().map(|&x| self.yes(x)).collect();
        let (sta, wsta) = self.states();
        self.known.truncate(self.known.len() - n);
        self.line("} else {");
        self.st = st0;
        self.wst = wst0;
        let n = self.push_known(np);
        b(self)?;
        let merge = |s: &Self, a: St, ya: bool, b: St| {
            let yb = s.yes(b);
            if a == b {
                a
            } else if ya && yb {
                St::Yes
            } else if ya {
                St::Cond(p)
            } else if yb {
                St::Cond(np)
            } else {
                St::Maybe
            }
        };
        let st: Vec<[St; 4]> = (0..self.st.len()).map(|k| std::array::from_fn(|c| merge(self, sta[k][c], ya[k][c], self.st[k][c]))).collect();
        let wst: Vec<St> = (0..self.wst.len()).map(|k| merge(self, wsta[k], wya[k], self.wst[k])).collect();
        self.known.truncate(self.known.len() - n);
        self.line("}");
        self.st = st;
        self.wst = wst;
        Ok(())
    }
    /// `if cond { f }` for a runtime validity check.
    fn guard(&mut self, cond: String, f: impl FnOnce(&mut Self) -> Result<(), String>) -> Result<(), String> {
        let id = self.preds.len();
        self.preds.push(Pred { name: cond, parent: None });
        self.under(id, f)
    }
    fn save(&self) -> Saved {
        Saved {
            st: self.st.clone(),
            wst: self.wst.clone(),
            lits: self.lits.len(),
            pinned: self.pinned.clone(),
            mk: self.mk.clone(),
            mkind: self.mkind.clone(),
            wk: self.wk.clone(),
            slot_next: self.slot_next.clone(),
            chains: self.chains.clone(),
            wchain: self.wchain.clone(),
            serial: self.serial,
            preds: self.preds.len(),
            negs: self.negs.clone(),
        }
    }
    fn restore(&mut self, s: &Saved) {
        self.st = s.st.clone();
        self.wst = s.wst.clone();
        self.lits.truncate(s.lits);
        self.pinned = s.pinned.clone();
        self.mk = s.mk.clone();
        self.mkind = s.mkind.clone();
        self.wk = s.wk.clone();
        self.slot_next = s.slot_next.clone();
        self.chains = s.chains.clone();
        self.wchain = s.wchain.clone();
        self.serial = s.serial;
        self.preds.truncate(s.preds);
        self.negs = s.negs.clone();
    }
    /// The state at a loop head or exit: two paths join. Predicates made inside the loop body
    /// do not hold outside one iteration.
    fn join(&self, a: &(Vec<[St; 4]>, Vec<St>), b: &(Vec<[St; 4]>, Vec<St>), npreds: usize) -> (Vec<[St; 4]>, Vec<St>) {
        let norm = |s: St| if matches!(s, St::Cond(q) if q >= npreds) { St::Maybe } else { s };
        let j = |x: St, y: St| {
            let (x, y) = (norm(x), norm(y));
            if x == y {
                x
            } else if x == St::Yes && self.yes(y) {
                y
            } else if y == St::Yes && self.yes(x) {
                x
            } else {
                St::Maybe
            }
        };
        let st = (0..a.0.len()).map(|k| std::array::from_fn(|c| j(a.0[k][c], b.0[k][c]))).collect();
        let wst = (0..a.1.len()).map(|k| j(a.1[k], b.1[k])).collect();
        (st, wst)
    }
    fn store_memo(&mut self, k: usize, c: usize, e: Expr) -> Result<(), String> {
        if self.mkind.get(&(k, c)).is_some_and(|&p| p != e.kind) {
            return Err(format!("PP register {k} kind differs between paths"));
        }
        self.mkind.insert((k, c), e.kind);
        self.decls.insert(format!("m{k}_{c}"), typename(e.kind));
        self.decls.insert(format!("v{k}_{c}"), "bool");
        self.line(&format!("m{k}_{c}={}; v{k}_{c}=true;", e.text));
        let prev = self.mk.get(&(k, c)).cloned().flatten();
        let merged = if self.mk.contains_key(&(k, c)) { kany(&[&prev, &e.konst]) } else { e.konst };
        self.mk.insert((k, c), merged);
        let name = format!("m{k}_{c}");
        let chain = merge_ids(self.chains.get(&name).map_or(&[][..], |c| &c[..]), &std::mem::take(&mut self.chain_out));
        if !chain.is_empty() {
            self.chains.insert(name, Rc::new(chain));
        }
        self.st[k][c] = St::Yes;
        Ok(())
    }
    fn assign(&mut self, k: usize, c: usize) -> Result<(), String> {
        let e = self.compute(k, c)?;
        self.store_memo(k, c, e)
    }
    /// Component c of register k, computed here unless it is known to be (Eval::comp).
    fn need(&mut self, k: usize, c: usize) -> Result<Expr, String> {
        if k >= self.f.ins.len() {
            return Ok(lit(0.));
        }
        let c = self.clamp(k, c);
        let s = self.st[k][c];
        if !self.yes(s) {
            if s == St::No {
                self.assign(k, c)?;
            } else {
                self.guard(format!("!v{k}_{c}"), |g| g.assign(k, c))?;
                self.st[k][c] = St::Yes;
            }
        }
        Ok(self.memo_expr(k, c))
    }
    /// The whole value of register k (`w<k>`), generated by `make` unless known to be computed.
    fn own_whole(&mut self, k: usize, make: impl FnOnce(&mut Self) -> Result<(Vec<K>, Vec<usize>), String>) -> Result<(), String> {
        if self.yes(self.wst[k]) {
            return Ok(());
        }
        self.decls.insert(format!("w{k}"), "vec4<u32>");
        self.decls.insert(format!("wv{k}"), "bool");
        let body = |s: &mut Self| -> Result<(), String> {
            let (ks, chain) = make(s)?;
            s.line(&format!("wv{k}=true;"));
            let prev = s.wchain.get(&k).cloned().unwrap_or_default();
            s.wchain.insert(k, merge_ids(&prev, &chain));
            let mut w: [K; 4] = Default::default();
            for (c, q) in ks.into_iter().take(4).enumerate() {
                w[c] = q;
            }
            let merged = match s.wk.get(&k) {
                Some(prev) => std::array::from_fn(|c| kany(&[&prev[c], &w[c]])),
                None => w,
            };
            s.wk.insert(k, merged);
            s.wst[k] = St::Yes;
            Ok(())
        };
        if self.wst[k] == St::No {
            body(self)?;
        } else {
            self.guard(format!("!wv{k}"), body)?;
        }
        self.wst[k] = St::Yes;
        Ok(())
    }
    /// Loops recompute these registers every iteration (Eval::invalidate).
    fn invalidate(&mut self, root: usize) {
        for r in self.f.gpu_invalidate(root) {
            let n = if comps(self.f.ins[r].ty) == 1 { 1 } else { 4 };
            let mut resets = Vec::new();
            for c in 0..n {
                self.decls.insert(format!("v{r}_{c}"), "bool");
                resets.push(format!("v{r}_{c}=false;"));
            }
            self.decls.insert(format!("wv{r}"), "bool");
            resets.push(format!("wv{r}=false;"));
            self.line(&resets.join(" "));
            self.st[r] = [St::No; 4];
            self.wst[r] = St::No;
        }
    }
    /// Component c of register r, demanded by a conditional's operand that the VM takes when
    /// predicate p holds: then it computes r (Eval::comp); otherwise it only materializes the
    /// operand's shared components (Eval::compute op 9, the untaken-branch rule). Both cases are
    /// emitted with one computation of each register where they coincide.
    fn branch(&mut self, r: usize, c: usize, p: usize) -> Result<Expr, String> {
        if r >= self.f.ins.len() {
            return Ok(lit(0.));
        }
        let c = self.clamp(r, c);
        let op = self.f.ins[r].op;
        if !self.volatile[r] {
            // no side effects: computed in either case (they are computed in advance)
            return self.need(r, c);
        }
        let hoisted = !self.nohoist && self.f.gpu_hoisted(r, c);
        if hoisted && !matches!(op, 6 | 7 | 11) {
            return self.need(r, c);
        }
        if self.yes(self.st[r][c]) {
            // the taken case reads the memo; the untaken one still visits the shared
            // components below it (normally computed already)
            if !hoisted && !self.nohoist {
                let found = self.f.gpu_shared(r, c);
                if !found.iter().all(|&(x, xc)| self.yes(self.st[x][xc])) {
                    let q = self.neg(p);
                    self.under(q, |s| found.into_iter().try_for_each(|(x, xc)| s.need(x, xc).map(|_| ())))?;
                }
            }
            return Ok(self.memo_expr(r, c));
        }
        if hoisted {
            // a shared setvar / write / loop does not run in an untaken branch
            self.under(p, |s| s.need(r, c).map(|_| ()))?;
            return Ok(self.memo_expr(r, c));
        }
        if self.st[r][c] == St::No && !self.f.gpu_forward(r) {
            if op == 9 {
                return self.branch_select(r, c, p);
            }
            // operations reading their operands in scalar_reads order, then computing
            if matches!(op, 7 | 13 | 15..=24 | 26..=47 | 48 | 49 | 50 | 53 | 54) && op != 25 {
                for (y, yc) in self.f.gpu_reads(r, c) {
                    self.branch(y, yc, p)?;
                }
                self.under(p, |s| s.need(r, c).map(|_| ()))?;
                return Ok(self.memo_expr(r, c));
            }
        }
        let found = if self.nohoist { vec![] } else { self.f.gpu_shared(r, c) };
        if found.is_empty() {
            self.under(p, |s| s.need(r, c).map(|_| ()))?;
        } else {
            self.under_else(p, |s| s.need(r, c).map(|_| ()), |s| found.into_iter().try_for_each(|(x, xc)| s.need(x, xc).map(|_| ())))?;
        }
        Ok(self.memo_expr(r, c))
    }
    /// A conditional reached through `branch` (not shared, not computed yet): its condition
    /// and operands with the predicates combined (when p fails, the VM visits all three).
    fn branch_select(&mut self, k: usize, c: usize, p: usize) -> Result<Expr, String> {
        let i = self.f.ins[k].clone();
        let arg = |j: usize| i.ops.get(j).copied().map(usize::from).ok_or_else(|| format!("PP op {} missing argument", i.op));
        let cond = self.branch(arg(0)?, 0, p)?;
        let ct = cast(&cond, Kind::B);
        let pa = self.pred(Some(p), &ct);
        let a = self.branch(arg(1)?, c, pa)?;
        let pb = self.pred(Some(p), &format!("!{ct}"));
        let b = self.branch(arg(2)?, c, pb)?;
        if a.kind != b.kind {
            return Err("PP mixed branch types".into());
        }
        let konst = self.select_konst(&cond, &a, &b);
        let chain = self.entries_of(&[&a, &b]);
        let e = Expr { text: format!("select({}, {}, {ct})", b.text, a.text), kind: a.kind, konst };
        self.under(p, |s| {
            s.chain_out = chain;
            s.store_memo(k, c, e)
        })?;
        Ok(self.memo_expr(k, c))
    }
    fn whole(&mut self,k:usize)->Result<(),String> {
        let Some(i)=self.f.ins.get(k) else {return Ok(())};
        let ty=i.ty;
        if !self.dynamic {
            for c in 0..comps(ty) {self.comp(k,c)?;}
            return Ok(());
        }
        self.own_whole(k,|s| {
            let mut values=Vec::new();let mut ks=Vec::new();let mut chain=Vec::new();
            for c in 0..comps(ty) {
                let e=castk(&s.comp(k,c)?,kind(ty));
                chain=merge_ids(&chain,&s.entries(&e));
                values.push(pack(&e));ks.push(e.konst);
            }
            values.resize(4,"0u".into());
            s.line(&format!("w{k}=vec4<u32>({});",values.join(",")));
            Ok((ks,chain))
        })
    }
    fn comp(&mut self,k:usize,c:usize)->Result<Expr,String> {
        if self.dynamic {self.need(k,c)} else {self.compute(k,c)}
    }
    fn compute(&mut self, k: usize, c: usize) -> Result<Expr, String> {
        let Some(i) = self.f.ins.get(k).cloned() else { return Ok(lit(0.)) };
        let c = if comps(i.ty) == 1 { 0 } else { c.min(3) };
        if let Some(e) = self.memo.get(&(k, c)) {
            return Ok(e.clone());
        }
        self.depth += 1;
        if self.depth > 256 {
            return Err("PP expression nesting limit".into());
        }
        let arg = |j: usize| i.ops.get(j).copied().map(usize::from).ok_or_else(|| format!("PP op {} missing argument", i.op));
        if !matches!(i.ty,1|4..=11) {return Err(format!("PP register {k} result type not ported"));}
        let tk = kind(i.ty);
        let literal = |word: usize| -> Result<u32, String> {
            let a = i.ops.len() % 2 + word * 2;
            Ok(*i.ops.get(a).ok_or("PP literal")? as u32 | (*i.ops.get(a + 1).ok_or("PP literal")? as u32) << 16)
        };
        // the per-node values that entered the result through arithmetic (see `arith`)
        let mut chain: Vec<usize> = Vec::new();
        let (text, rk, konst): (String, Kind, K) = match i.op {
            0 if i.ty == 1 => {
                let e = self.value(u32::from(i.ops.first().is_none_or(|&v| v != 0)), Kind::B);
                (e.text, Kind::B, e.konst)
            }
            0 => {
                let word = literal(c.min(comps(i.ty) - 1))?;
                let e = self.value(word, if tk == Kind::I { Kind::I } else { Kind::F });
                (e.text, tk, e.konst)
            }
            1 => {
                let s = arg(0)?;
                let t = match (s, c) {
                    (8, 0) => "pos.x",
                    (8, 1) => "pos.y",
                    (1, 0) => "f32(config.header.z)",
                    (1, 1) => "f32(config.header.y/config.header.z)",
                    (3, 0) => "f32(firstLeadingBit(config.header.z))",
                    (3, 1) => "f32(firstLeadingBit(config.header.y/config.header.z))",
                    _ => "0.0",
                };
                let x = Expr { text: t.into(), kind: Kind::F, konst: if t == "0.0" { kconst() } else { None } };
                let e = castk(&x, tk);
                (e.text, tk, e.konst)
            }
            2 | 3 => {
                let v = if i.op == 2 { (self.input)(literal(0)?) } else { (self.vp)(arg(0)?) };
                let x = self.component_value(v, c);
                let e = castk(&x, tk);
                (e.text, tk, e.konst)
            }
            4 if self.dynamic => {
                let n=comps(i.ty); let base=arg(0)? as u16;
                let slots:Vec<u16>=(0..n).map(|j|base.wrapping_add(j as u16)).collect();
                let ks:Vec<K>=slots.iter().map(|&s|self.slot_konst(s).0).collect();
                let vchain=slots.iter().fold(Vec::new(),|acc,&s|merge_ids(&acc,&self.slot_konst(s).1));
                let ids:Vec<usize>=slots.iter().map(|s|self.var_ids[s]).collect();
                self.own_whole(k,|s| {
                    let mut values:Vec<String>=ids.iter().map(|idx|pack(&rt(format!("var_{}({idx}u)",typename(tk)),tk))).collect();
                    values.resize(4,"0u".into());
                    s.line(&format!("w{k}=vec4<u32>({});",values.join(",")));
                    Ok((ks,vchain))
                })?;
                let e=self.whole_comp(k,c,tk);chain=self.entries(&e);(e.text,tk,e.konst)
            }
            4 => {
                let n = comps(i.ty);
                if !self.reads.contains_key(&k) {
                    // getvar snapshots the whole vector on its first component demand.
                    let base = arg(0)? as u16;
                    let mut snapshot = Vec::new();
                    for j in 0..n {
                        let slot = base.wrapping_add(j as u16);
                        if !self.vars.contains_key(&slot) {
                            // the state left by the size/seed functions, read on first use
                            if let Some(&(bits, kd)) = self.initial.get(&slot) {
                                let e = self.value(bits, kd);
                                self.vars.insert(slot, e);
                            }
                        }
                        snapshot.push(self.vars.get(&slot).cloned().unwrap_or_else(|| lit(0.)));
                    }
                    self.reads.insert(k, snapshot);
                }
                let x = self.reads[&k].get(c).cloned().unwrap_or_else(|| lit(0.));
                let e = castk(&x, tk);
                chain = self.entries(&e);
                (e.text, tk, e.konst)
            }
            7 if self.dynamic => {
                let x=self.comp(arg(0)?,c)?;
                let slot=(arg(1)? as u16).wrapping_add(c.min(comps(i.ty)-1) as u16);
                let idx=self.var_ids[&slot];
                let v=castk(&x,tk);
                self.line(&format!("variables[{idx}u]=vec2<u32>({},{}u);",pack(&v),tk as u32));
                let prev=self.slot_next.get(&slot).cloned().unwrap_or_else(||(kconst(),vec![]));
                let stored=merge_ids(&prev.1,&self.entries(&v));
                self.slot_next.insert(slot,(kany(&[&prev.0,&v.konst]),stored));
                chain=self.entries(&x);
                (x.text,x.kind,x.konst)
            }
            7 => {
                let x = self.comp(arg(0)?, c)?;
                let slot = (arg(1)? as u16).wrapping_add(c.min(comps(i.ty) - 1) as u16);
                self.vars.insert(slot, castk(&x, tk));
                chain = self.entries(&x);
                (x.text, x.kind, x.konst)
            }
            6 => {
                let a = arg(0)?;
                self.whole(a)?;
                let at = self.f.ins.get(a).map_or(4, |i| i.ty);
                let n = comps(at);
                let x = if n > 1 && c >= n {lit(0.)} else if self.dynamic {
                    self.whole_comp(a,if n==1 {0} else {c},kind(at))
                } else { self.comp(a, c)? };
                let e = castk(&x, kind(at));
                chain = self.entries(&e);
                (e.text, kind(at), e.konst)
            }
            9 if self.dynamic => {
                let cond=self.comp(arg(0)?,0)?;
                let ct=cast(&cond,Kind::B);
                let pa=self.pred(None,&ct);
                let a=self.branch(arg(1)?,c,pa)?;
                let pb=self.neg(pa);
                let b=self.branch(arg(2)?,c,pb)?;
                if a.kind!=b.kind {return Err("PP mixed branch types".into());}
                let konst=self.select_konst(&cond,&a,&b);
                chain=self.entries_of(&[&a,&b]);
                (format!("select({}, {}, {})",b.text,a.text,self.preds[pa].name),a.kind,konst)
            }
            9 => {
                let cond = self.comp(arg(0)?, 0)?;
                let a = self.comp(arg(1)?, c)?;
                let b = self.comp(arg(2)?, c)?;
                if a.kind != b.kind {
                    return Err("PP mixed branch types".into());
                }
                let konst = self.select_konst(&cond, &a, &b);
                chain = self.entries_of(&[&a, &b]);
                (format!("select({}, {}, {})", b.text, a.text, cast(&cond, Kind::B)), a.kind, konst)
            }
            11 if self.dynamic => {
                let init=arg(0)?;let cond=arg(1)?;let body=arg(2)?;
                let ty=self.f.ins.get(init).ok_or("PP loop init")?.ty;
                if self.f.ins.get(body).ok_or("PP loop body")?.ty!=ty || i.ty!=ty {return Err("PP mixed loop value types".into());}
                let tail=&i.ops[3..];let start=tail.len()%2;
                let max=*tail.get(start).ok_or("PP loop limit")? as u32 | (*tail.get(start+1).ok_or("PP loop limit")? as u32)<<16;
                let limit=if max==0 {1<<24} else {max.min(1<<24)};
                self.own_whole(k,|s| s.emit_loop(k,init,cond,body,ty,limit))?;
                let e=self.whole_comp(k,c,tk);chain=self.entries(&e);(e.text,tk,e.konst)
            }
            12 => {
                let a = arg(0)?;
                let b = arg(1)?;
                if self.side.get(a) == Some(&true) {
                    self.whole(a)?;
                }
                if self.side.get(b) == Some(&true) {
                    self.whole(b)?;
                }
                let x = self.comp(b, c)?;
                chain = self.entries(&x);
                (x.text, x.kind, x.konst)
            }
            13 => {
                let a = arg(0)?;
                let b = arg(1)?;
                let na = self.f.ins.get(a).map_or(0, |i| comps(i.ty));
                let x = if c < na { self.comp(a, c)? } else { self.comp(b, c - na)? };
                let e = castk(&x, tk);
                chain = self.entries(&e);
                (e.text, tk, e.konst)
            }
            15 => {
                let x = self.comp(arg(c)?, 0)?;
                let e = castk(&x, tk);
                chain = self.entries(&e);
                (e.text, tk, e.konst)
            }
            16 => {
                let x = self.comp(arg(0)?, (arg(1)? >> (2 * c)) & 3)?;
                chain = self.entries(&x);
                (x.text, x.kind, x.konst)
            }
            17 => {
                let x = self.comp(arg(0)?, c)?;
                if tk == Kind::I && x.kind == Kind::F {
                    let konst = kjoin(&[&x.konst], true);
                    (format!("to_int(floor({}))", x.text), tk, konst)
                } else {
                    let e = castk(&x, tk);
                    if e.kind == Kind::F && x.kind == Kind::F {
                        chain = self.entries(&e);
                    }
                    (e.text, tk, e.konst)
                }
            }
            18..=22 | 48 | 49 | 54 => {
                let mut vals = Vec::new();
                for r in [arg(0)?, arg(1)?] {
                    let zero = i.op != 20 && c > 0 && comps(i.ty) > 1 && self.f.ins.get(r).is_some_and(|i| comps(i.ty) == 1);
                    let v = if zero {
                        let kd = kind(self.f.ins[r].ty);
                        Expr { text: if kd == Kind::I { "0i" } else { "0.0" }.into(), kind: kd, konst: kconst() }
                    } else {
                        self.comp(r, c)?
                    };
                    vals.push(v);
                }
                let rk = if tk == Kind::I || vals.iter().all(|e| e.kind == Kind::I) { Kind::I } else { Kind::F };
                let (ea, eb) = (castk(&vals[0], rk), castk(&vals[1], rk));
                let (a, b) = (&ea.text, &eb.text);
                let t = match i.op {
                    18 => format!("({a}+{b})"),
                    19 => format!("({a}-{b})"),
                    20 => format!("({a}*{b})"),
                    21 if rk == Kind::I => format!("int_div({a},{b})"),
                    21 => format!("({a}/{b})"),
                    22 => format!("{}_mod({a},{b})", if rk == Kind::I { "int" } else { "float" }),
                    48 => format!("min({a},{b})"),
                    49 => format!("max({a},{b})"),
                    _ if rk == Kind::I => format!("int_pow({a},{b})"),
                    _ => format!("pp_pow({a},{b})"),
                };
                let float = rk == Kind::F;
                let (safe, uses) = match i.op {
                    54 => (false, [Use::Exact, Use::Exact]),
                    // a known divisor or dividend changes the instructions chosen: dirtyconcrete
                    // node 599, (x + 1) / 2, and test-2-twm node 502, 0.5 / sqrt(x), differ in
                    // the last bit with the value read at run time
                    21 | 22 if float => (false, [Use::Exact, Use::Exact]),
                    // the driver folds chains of known float operations with other roundings
                    // (test-2-twm node 502: (a*b)*c), so a known float result stays embedded
                    18..=20 if float => (false, [Use::Plain, Use::Plain]),
                    _ => (true, [Use::Plain, Use::Plain]),
                };
                let konst = kjoin(&[&ea.konst, &eb.konst], safe);
                let konst = self.finish(konst, &[(&ea, uses[0]), (&eb, uses[1])]);
                if float && konst.is_none() && matches!(i.op, 18..=22) {
                    // a division by a known value is a product for the driver (stripe-new-1
                    // node 138: (x - a) / 2)
                    chain = self.arith(&[&ea, &eb]);
                }
                (t, rk, konst)
            }
            23 => {
                let x = self.comp(arg(0)?, c)?;
                let k = if x.kind == Kind::I { Kind::I } else { Kind::F };
                let e = castk(&x, k);
                // negation is exact
                let konst = kjoin(&[&e.konst], true);
                if k == Kind::F && konst.is_none() {
                    chain = self.arith(&[&e]);
                }
                (format!("(-{})", e.text), k, konst)
            }
            24 => {
                let a = arg(0)?;
                let b = arg(1)?;
                let ai = self.f.ins.get(a).ok_or("dot operand")?;
                let bi = self.f.ins.get(b).ok_or("dot operand")?;
                let n = comps(ai.ty).min(comps(bi.ty)).max(1);
                let rk = if tk == Kind::I && kind(ai.ty) == Kind::I && kind(bi.ty) == Kind::I { Kind::I } else { Kind::F };
                let mut sum = if rk == Kind::I { "0i" } else { "0.0" }.to_string();
                let mut terms = Vec::new();
                for j in 0..n {
                    let x = castk(&self.comp(a, j)?, rk);
                    let y = castk(&self.comp(b, j)?, rk);
                    sum = format!("({sum}+({}*{}))", x.text, y.text);
                    terms.push((x, y));
                }
                let all: Vec<&K> = terms.iter().flat_map(|(x, y)| [&x.konst, &y.konst]).collect();
                let konst = kjoin(&all, rk == Kind::I);
                if konst.is_none() {
                    // a product of two known values would be folded before the addition
                    for (x, y) in &terms {
                        let u = if x.konst.is_some() && y.konst.is_some() { Use::Exact } else { Use::Plain };
                        self.pin_use(x, u);
                        self.pin_use(y, u);
                    }
                    if rk == Kind::F {
                        let ops: Vec<&Expr> = terms.iter().flat_map(|(x, y)| [x, y]).collect();
                        chain = self.arith(&ops);
                    }
                }
                let e = castk(&Expr { text: sum, kind: rk, konst }, tk);
                (e.text, tk, e.konst)
            }
            26..=34 => {
                let a = self.comp(arg(0)?, 0)?;
                if i.op == 28 {
                    let e = castk(&a, Kind::B);
                    (format!("(!{})", e.text), Kind::B, e.konst)
                } else {
                    let b = self.comp(arg(1)?, 0)?;
                    let rk = if i.op <= 27 {
                        Kind::B
                    } else if a.kind == Kind::I && b.kind == Kind::I {
                        Kind::I
                    } else {
                        Kind::F
                    };
                    let op = match i.op {
                        26 => "&&",
                        27 => "||",
                        29 => "==",
                        30 => "!=",
                        31 => ">",
                        32 => ">=",
                        33 => "<",
                        _ => "<=",
                    };
                    let (ea, eb) = (castk(&a, rk), castk(&b, rk));
                    let konst = kjoin(&[&ea.konst, &eb.konst], true);
                    let konst = self.finish(konst, &[(&ea, Use::Plain), (&eb, Use::Plain)]);
                    (format!("({} {op} {})", ea.text, eb.text), Kind::B, konst)
                }
            }
            35..=44 | 53 => {
                let a = self.comp(arg(0)?, c)?;
                if i.op == 35 && a.kind == Kind::I {
                    (format!("abs({})", a.text), Kind::I, kjoin(&[&a.konst], true))
                } else {
                    let fun = match i.op {
                        35 => "abs",
                        36 => "floor",
                        37 => "ceil",
                        38 => "cos",
                        39 => "sin",
                        40 => "pp_sqrt",
                        41 => "log",
                        42 => "exp",
                        43 => "exp2",
                        44 => "tan",
                        _ => "log2",
                    };
                    let x = castk(&a, Kind::F);
                    let konst = kjoin(&[&x.konst], matches!(i.op, 35..=37));
                    let e = castk(&Expr { text: format!("{fun}({})", x.text), kind: Kind::F, konst }, tk);
                    (e.text, tk, e.konst)
                }
            }
            45 => {
                let a = self.comp(arg(0)?, 0)?;
                let b = self.comp(arg(0)?, 1)?;
                let (x, y) = (castk(&a, Kind::F), castk(&b, Kind::F));
                let konst = kjoin(&[&x.konst, &y.konst], false);
                (format!("atan2({},{})", y.text, x.text), Kind::F, konst)
            }
            46 => {
                let a = self.comp(arg(0)?, 0)?;
                let b = self.comp(arg(1)?, 0)?;
                let (rho, th) = (castk(&a, Kind::F), castk(&b, Kind::F));
                let konst = kjoin(&[&rho.konst, &th.konst], false);
                let konst = self.finish(konst, &[(&rho, Use::Plain), (&th, Use::Exact)]);
                if konst.is_none() {
                    chain = self.arith(&[&rho]);
                }
                (format!("({}*{}({}))", rho.text, if c == 0 { "cos" } else { "sin" }, th.text), Kind::F, konst)
            }
            47 => {
                let a = self.comp(arg(0)?, c)?;
                let b = self.comp(arg(1)?, c)?;
                let t = self.comp(arg(2)?, c)?;
                let a = castk(&a, Kind::F);
                let b = castk(&b, Kind::F);
                let t = castk(&t, Kind::F);
                let konst = kjoin(&[&a.konst, &b.konst, &t.konst], false);
                let konst = self.finish(konst, &[(&a, Use::Plain), (&b, Use::Plain), (&t, Use::Plain)]);
                if konst.is_none() {
                    chain = self.arith(&[&a, &b, &t]);
                }
                let (a, b, t) = (&a.text, &b.text, &t.text);
                (format!("({a}+({b}-{a})*{t})"), Kind::F, konst)
            }
            50 => {
                let a = self.comp(arg(0)?, c)?;
                let x = castk(&a, Kind::F);
                self.pin_use(&x, Use::Plain);
                chain = self.arith(&[&x]);
                (format!("({}*random_draw())", x.text), Kind::F, None)
            }
            51 | 52 if self.dynamic => {
                let slot=arg(1)?;let filter=i.ops.get(2).copied().unwrap_or(0);let pt=arg(0)?;
                self.own_whole(k,|s| {
                    let x=castk(&s.comp(pt,0)?,Kind::F);let y=castk(&s.comp(pt,1)?,Kind::F);
                    s.pin_use(&x,Use::Plain);s.pin_use(&y,Use::Plain);
                    s.line(&format!("w{k}=bitcast<vec4<u32>>(sample_image({slot}u,vec2<f32>({},{}),{filter}u,{}));",x.text,y.text,i.op==52));
                    Ok((vec![],vec![]))
                })?;
                let x=self.whole_comp(k,if i.op==51 {0} else {c},Kind::F);(x.text,Kind::F,None)
            }
            51 | 52 => {
                let sample = if let Some(s) = self.samples.get(&k) {
                    s.clone()
                } else {
                    let x = castk(&self.comp(arg(0)?, 0)?, Kind::F);
                    let y = castk(&self.comp(arg(0)?, 1)?, Kind::F);
                    self.pin_use(&x, Use::Plain);
                    self.pin_use(&y, Use::Plain);
                    let slot = arg(1)?;
                    let filter = i.ops.get(2).copied().unwrap_or(0);
                    let name = format!("sample{k}");
                    self.body.push_str(&format!(
                        "    let {name}=sample_image({slot}u,vec2<f32>({},{}),{filter}u,{});\n",
                        x.text,
                        y.text,
                        i.op == 52
                    ));
                    self.samples.insert(k, name.clone());
                    name
                };
                (format!("{sample}[{}u]", if i.op == 51 { 0 } else { c }), Kind::F, None)
            }
            _ => return Err(format!("PP opcode {} not ported", i.op)),
        };
        self.depth -= 1;
        if self.dynamic {
            self.chain_out = chain;
            return Ok(Expr { text, kind: rk, konst });
        }
        let name = format!("r{k}_{c}");
        self.body.push_str(&format!(
            "    let {name}:{}={text};\n",
            match rk {
                Kind::F => "f32",
                Kind::I => "i32",
                Kind::B => "bool",
            }
        ));
        if !chain.is_empty() {
            self.chains.insert(name.clone(), Rc::new(chain));
        }
        let e = Expr { text: name, kind: rk, konst };
        self.memo.insert((k, c), e.clone());
        Ok(e)
    }
    /// A loop's whole value (Eval::compute op 11): the body is generated from the state at the
    /// loop head, which joins the state before the loop with the state after an iteration, so
    /// it is regenerated until that state no longer changes.
    fn emit_loop(&mut self, k: usize, init: usize, cond: usize, body: usize, ty: u8, limit: u32) -> Result<(Vec<K>, Vec<usize>), String> {
        self.whole(init)?;
        let ik: [K; 4] = self.wk.get(&init).cloned().unwrap_or_default();
        let init_chain = self.wchain.get(&init).cloned().unwrap_or_default();
        for q in &ik {
            self.pin_use(&Expr { text: String::new(), kind: Kind::F, konst: q.clone() }, Use::Exact);
        }
        let serial = self.serial;
        self.serial += 1;
        let start = if init < self.f.ins.len() { format!("w{init}") } else { "vec4<u32>(0u)".into() };
        self.line(&format!("var last{serial}={start};"));
        self.line(&format!("var iterations{serial}=0u;"));
        self.line("loop {");
        let mark = self.body.len();
        let saved = self.save();
        let npreds = self.preds.len();
        let mut entry = (saved.st.clone(), saved.wst.clone());
        for _ in 0..64 {
            self.body.truncate(mark);
            self.restore(&saved);
            self.st = entry.0.clone();
            self.wst = entry.1.clone();
            self.invalidate(cond);
            self.invalidate(body);
            let condition = self.comp(cond, 0)?;
            self.pin_use(&condition, Use::Exact);
            self.line(&format!("if {} {{break;}}", cast(&condition, Kind::B)));
            let at_cond = self.states();
            let mut vals = Vec::new();
            let mut ks = Vec::new();
            let mut chain = init_chain.clone();
            for j in 0..comps(ty) {
                let x = castk(&self.comp(body, j)?, kind(ty));
                vals.push(pack(&x));
                ks.push(kany(&[&x.konst, &ik[j.min(3)]]));
                chain = merge_ids(&chain, &self.entries(&x));
            }
            vals.resize(4, "0u".into());
            self.line(&format!("last{serial}=vec4<u32>({});", vals.join(",")));
            self.line(&format!("iterations{serial}+=1u;"));
            self.line(&format!("if iterations{serial}>={limit}u {{break;}}"));
            let at_end = self.states();
            let next = self.join(&entry, &at_end, npreds);
            if next == entry {
                let after = self.join(&at_cond, &at_end, npreds);
                self.st = after.0;
                self.wst = after.1;
                self.line("}");
                self.line(&format!("w{k}=last{serial};"));
                return Ok((ks, chain));
            }
            entry = next;
        }
        Err("PP loop state does not settle".into())
    }

    /// The text with every value token replaced: embedded (pinned, as before) or read from
    /// `config.lits[index]`.
    fn resolve(&self, text: &str, index: &[Option<usize>]) -> String {
        let mut out = String::with_capacity(text.len());
        let mut parts = text.split(TOKEN);
        out.push_str(parts.next().unwrap_or(""));
        while let (Some(id), Some(rest)) = (parts.next(), parts.next()) {
            let id: usize = id.parse().expect("value token");
            let (bits, k) = self.lits[id];
            match (index[id], k) {
                (None, Some(Kind::F)) => out.push_str(&format!("bitcast<f32>(0x{bits:08x}u)")),
                (None, Some(Kind::I)) => out.push_str(&format!("bitcast<i32>(0x{bits:08x}u)")),
                (None, Some(Kind::B)) => out.push_str(if bits != 0 { "true" } else { "false" }),
                (None, None) => out.push_str(&format!("0x{bits:08x}u")),
                (Some(n), Some(Kind::F)) => out.push_str(&format!("bitcast<f32>(config.lits[{n}u])")),
                (Some(n), Some(Kind::I)) => out.push_str(&format!("bitcast<i32>(config.lits[{n}u])")),
                (Some(n), Some(Kind::B)) => out.push_str(&format!("(config.lits[{n}u]!=0u)")),
                (Some(n), None) => out.push_str(&format!("config.lits[{n}u]")),
            }
            out.push_str(rest);
        }
        out
    }
}

/// A generated PP shader: its WGSL (the pipeline cache key) and the per-node values the shader
/// reads from `config.lits` (appended to the 40 Config words at dispatch).
pub struct Shader {
    pub text: String,
    pub lits: Vec<u32>,
    /// the control-flow form
    pub flow: bool,
    /// value ids (generation order) read from the buffer, in buffer order (research dumps)
    pub moved: Vec<usize>,
}

pub fn compile(f: &Func, input: &dyn Fn(u32) -> Value, vp: &dyn Fn(usize) -> Value, vars: &[(u16, Value)]) -> Result<Shader, String> {
    if f.ins.is_empty() || f.ins.len() > 4096 {
        return Err("PP shader size limit".into());
    }
    let dynamic=f.ins.iter().any(|i|i.op==11 && i.ty!=0) || (f.ins.iter().any(|i|i.op==9) && f.ins.iter().any(|i|matches!(i.op,7|50))) || crate::envc::var("SDRT_GPU_DYNAMIC").is_ok();
    if f.ins.iter().any(|i|i.op==11 && i.ty!=0) && crate::envc::var("SDRT_LOOPDO").is_ok_and(|v|v!="until") {return Err("GPU supports the default until-loop mode".into());}
    let initial:HashMap<_,_>=vars.iter().map(|&(slot,v)|(slot,component(v,0))).collect();
    let mut slots=std::collections::BTreeSet::new();
    for i in &f.ins {
        if matches!(i.op,4|7) {
            let base=*i.ops.get(if i.op==4 {0} else {1}).ok_or("PP variable slot")?;
            for c in 0..comps(i.ty) {slots.insert(base.wrapping_add(c as u16));}
        }
    }
    let var_ids:HashMap<_,_>=slots.into_iter().enumerate().map(|(idx,slot)|(slot,idx)).collect();
    // Control-flow form: what the driver may know of each variable slot depends on the stores,
    // which depend on what it knows of the slots read: passes until it settles.
    let mut slot_k: HashMap<u16, Slot> = HashMap::new();
    for pass in 0..32 {
        let n=f.ins.len();
        let mut g=Compiler {f,input,vp,vars:HashMap::new(),initial:initial.clone(),lits:Vec::new(),pinned:HashSet::new(),reads:HashMap::new(),memo:HashMap::new(),samples:HashMap::new(),side:f.gpu_side().to_vec(),body:String::new(),depth:0,dynamic,serial:0,var_ids:var_ids.clone(),
            volatile:f.gpu_volatile(),nohoist:crate::envc::var("SDRT_NOHOIST").is_ok(),st:vec![[St::No;4];n],wst:vec![St::No;n],preds:Vec::new(),negs:HashMap::new(),known:Vec::new(),decls:BTreeMap::new(),mk:HashMap::new(),mkind:HashMap::new(),wk:HashMap::new(),slot_k:slot_k.clone(),slot_next:HashMap::new(),chains:HashMap::new(),wchain:HashMap::new(),chain_out:Vec::new()};
        let mut initial_lines=String::new();
        if dynamic {
            // the variable state after the size/seed functions, slot by slot in index order
            let mut ids:Vec<_>=g.var_ids.iter().map(|(&slot,&idx)|(idx,slot)).collect();ids.sort();
            for (idx,slot) in ids {
                if let Some(&(bits,k))=g.initial.get(&slot) {
                    let id=g.lits.len();
                    g.lits.push((bits,None));
                    initial_lines.push_str(&format!("    variables[{idx}u]=vec2<u32>({TOKEN}{id}{TOKEN},{}u);\n",k as u32));
                    g.slot_next.insert(slot,(Some(Rc::new(Konst {deps:vec![id],safe:true})),vec![]));
                }
            }
            // registers without side effects or state, computed in advance in register order:
            // the order does not matter and each is computed once
            let mut demand=vec![0u8;n];
            demand[n-1]=((1u32<<comps(f.ins[n-1].ty).min(4))-1) as u8;
            for k in (0..n).rev() {
                for c in 0..4 {
                    if demand[k]&(1<<c)==0 {continue;}
                    let i=&f.ins[k];
                    for (r,rc) in f.gpu_reads(k,c) {
                        // a sequence computes its first operand only for its side effects
                        if i.op==12 && Some(&(r as u16))==i.ops.first() && !g.side[r] {continue;}
                        demand[r]|=1<<rc;
                    }
                }
            }
            for k in 0..n {
                if g.volatile[k] {continue;}
                for c in 0..4 {
                    if demand[k]&(1<<c)!=0 {g.need(k,c)?;}
                }
            }
        }
        let last = n - 1;
        let nc = comps(f.ins[last].ty);
        let mut cs = Vec::new();
        for c in 0..nc {
            let e = g.comp(last, c)?;
            cs.push(castk(&e, kind(f.ins[last].ty)));
        }
        // Eval::value first packs the declared result type; the pixel output then converts to float.
        let cs: Vec<Expr> = cs.into_iter().map(|e| castk(&e, Kind::F)).collect();
        for e in &cs {
            g.pin_use(e, Use::Plain);
        }
        let cs: Vec<String> = cs.into_iter().map(|e| e.text).collect();
        if dynamic {
            // the stores' knowledge, with the initial state of every slot read
            let mut next=g.slot_next.clone();
            for &slot in g.var_ids.keys() {
                next.entry(slot).or_insert_with(||(kconst(),vec![]));
            }
            if next!=slot_k {
                if pass<31 {
                    slot_k=next;
                    continue;
                }
                // not settled (never seen): every value embedded, as before the buffer
                g.pinned.extend(0..g.lits.len());
            }
        }
        let color = if nc >= 4 {
            format!("vec4<f32>({})", cs.join(","))
        } else if nc == 3 {
            format!("vec4<f32>({},1.0)", cs.join(","))
        } else {
            format!("vec4<f32>({0},{0},{0},1.0)", cs[0])
        };
        if let Ok(spec) = crate::envc::var("SDRT_GPU_PIN") {
            // Research switch. "1"/"all": every value embedded (each node's text as before this
            // buffer); "none": every value read from the buffer (results change, see REPORT);
            // "N:all", "N:+i,j" (also embed values i, j), "N:-i,j" (embed all but i, j) for PP
            // node N only.
            let node = NODE.with(|n| n.get());
            if spec == "none" {
                g.pinned.clear();
            }
            let (target, list) = match spec.split_once(':') {
                Some((n, l)) => (n.parse::<usize>().ok(), l.to_string()),
                None => (None, "all".to_string()),
            };
            if spec != "none" && target.is_none_or(|t| t == node) {
                let ids = |l: &str| l.split(',').filter_map(|x| x.trim().parse::<usize>().ok()).collect::<Vec<_>>();
                if let Some(l) = list.strip_prefix('+') {
                    g.pinned.extend(ids(l));
                } else if let Some(l) = list.strip_prefix('-') {
                    let keep = ids(l);
                    g.pinned.extend((0..g.lits.len()).filter(|i| !keep.contains(i)));
                } else {
                    g.pinned.extend(0..g.lits.len());
                }
            }
        }
        let mut index=vec![None;g.lits.len()];
        let mut lits=Vec::new();let mut moved=Vec::new();
        for (id,slot) in index.iter_mut().enumerate() {
            if !g.pinned.contains(&id) {*slot=Some(lits.len());lits.push(g.lits[id].0);moved.push(id);}
        }
        let decls:String=g.decls.iter().map(|(name,ty)|format!("    var {name}:{ty};\n")).collect();
        let body=g.resolve(&g.body,&index);
        let initial_lines=g.resolve(&initial_lines,&index);
        // (the initial variable lines depend on the seed function's variables: not counted, so
        // the preflight and the post-seed compile agree)
        let flow_lines=if dynamic {decls.lines().count()+body.lines().count()} else {0};
        if dynamic {
            // Each register component is computed in one place; copies come only from runtime
            // checks of components computed on some paths (sequences of whole values below
            // such checks: stripe-new-1 node 158, 3,364 instructions, 141,856 lines, 256 s in
            // the driver). Creation time grows about quadratically: 8,350 lines take 1 s.
            const FLOW_LIMIT: usize = 20_000;
            let limit = crate::envc::var("SDRT_GPU_FLOW_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(FLOW_LIMIT);
            if crate::envc::var("SDRT_GPU_LOG").is_ok() {
                eprintln!("GPU PP control flow: {flow_lines} WGSL lines");
            }
            if flow_lines > limit {
                return Err(format!("PP control-flow shader too large ({flow_lines} lines)"));
            }
        }
        let prelude = include_str!("gpu_prelude.wgsl").split("@compute").next().unwrap();
        // PP shaders read their per-node values from words 40.. of the Config binding
        let at = prelude.find("struct Config {").and_then(|s| prelude[s..].find("\n}").map(|e| s + e)).ok_or("prelude without Config")?;
        let base = format!("{}\n    lits: array<u32>, // PP per-node values (gpu_pp.rs Shader::lits){}", &prelude[..at], &prelude[at..]);
        let control=if dynamic {format!("var<private> variables:array<vec2<u32>,{}>;\n{}",g.var_ids.len().max(1),include_str!("gpu_control.wgsl"))} else {String::new()};
        let text=format!("{base}\n{}\n{control}\n@compute @workgroup_size(8,8)\nfn main(@builtin(global_invocation_id) id:vec3<u32>) {{\n    let k=id.y*config.header.z+id.x;\n    if id.x>=config.header.z || k>=config.header.y {{return;}}\n    let pos=(vec2<f32>(id.xy)+vec2<f32>(0.5))/vec2<f32>(f32(config.header.z),f32(config.header.y/config.header.z));\n    rng=pixel_seed(config.header.w,pos);\n{decls}{initial_lines}{body}    store(k,{color});\n}}\n",include_str!("gpu_pp.wgsl"));
        return Ok(Shader {text,lits,flow:dynamic,moved});
    }
    unreachable!("the last pass returns")
}
