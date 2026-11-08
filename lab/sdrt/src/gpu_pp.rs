//! Scalar-demand PP code generation. The traversal order mirrors vm::Eval::comp, including
//! component memoization and rand draws. Unsupported control flow stays on the CPU as a whole.
use crate::vm::{Func, Value};
use std::collections::HashMap;
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    F,
    I,
    B,
}
#[derive(Clone)]
struct Expr {
    text: String,
    kind: Kind,
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
fn unpack(text:String,k:Kind)->Expr {Expr {text:if k==Kind::B {format!("({text}!=0u)")} else {format!("bitcast<{}>({text})",typename(k))},kind:k}}
fn comps(ty: u8) -> usize {
    match ty {
        4..=7 => (ty - 3) as usize,
        8..=11 => (ty - 7) as usize,
        _ => 1,
    }
}
fn lit(x: f32) -> Expr {
    Expr { text: format!("bitcast<f32>(0x{:08x}u)", x.to_bits()), kind: Kind::F }
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
fn component(v: Value, c: usize) -> Expr {
    let c = if v.n() <= 1 { 0 } else { c };
    match v {
        Value::I(n, a) => Expr { text: format!("bitcast<i32>(0x{:08x}u)", if c < n as usize { a[c] as u32 } else { 0 }), kind: Kind::I },
        Value::B(b) => Expr { text: b.to_string(), kind: Kind::B },
        Value::F(n, a) => lit(if c < n as usize { a[c] } else { 0. }),
        _ => lit(0.),
    }
}
struct Compiler<'a> {
    f: &'a Func,
    input: &'a dyn Fn(u32) -> Value,
    vp: &'a dyn Fn(usize) -> Value,
    vars: HashMap<u16, Expr>,
    reads: HashMap<usize, Vec<Expr>>,
    memo: HashMap<(usize, usize), Expr>,
    samples: HashMap<usize, String>,
    side: Vec<bool>,
    body: String,
    depth: usize,
    dynamic: bool,
    functions: String,
    serial: usize,
    var_ids: HashMap<u16,usize>,
}
impl Compiler<'_> {
    fn whole(&mut self,k:usize)->Result<(),String> {
        let Some(i)=self.f.ins.get(k) else {return Ok(())};
        let ty=i.ty;
        if self.dynamic {self.body.push_str(&format!("    if whole_valid[{k}u]==0u {{\n"));}
        let mut values=Vec::new();
        for c in 0..comps(ty) {
            let e=self.comp(k,c)?;
            values.push(pack(&Expr {text:cast(&e,kind(ty)),kind:kind(ty)}));
        }
        if self.dynamic {
            values.resize(4,"0u".into());
            self.body.push_str(&format!("    whole_bits[{k}u]=vec4<u32>({}); whole_valid[{k}u]=1u;\n    }}\n",values.join(",")));
        }
        Ok(())
    }
    fn comp(&mut self,k:usize,c:usize)->Result<Expr,String> {
        if !self.dynamic {return self.compute(k,c);}
        let Some(i)=self.f.ins.get(k) else {return Ok(lit(0.))};
        let c=if comps(i.ty)==1 {0} else {c.min(3)};
        if !self.memo.contains_key(&(k,c)) {
            let parent=std::mem::take(&mut self.body);
            let result=self.compute(k,c)?;
            let code=std::mem::replace(&mut self.body,parent);
            let cached=unpack(format!("memo_bits[{k}u][{c}u]"),result.kind);
            self.functions.push_str(&format!("fn eval_{k}_{c}(pos:vec2<f32>)->{} {{\n    if (memo_valid[{k}u]&{}u)!=0u {{return {};}}\n{code}    memo_bits[{k}u][{c}u]={}; memo_valid[{k}u]|={}u;\n    return {};\n}}\n",typename(result.kind),1<<c,cached.text,pack(&result),1<<c,result.text));
        }
        let rk=self.memo[&(k,c)].kind;
        let name=format!("t{}",self.serial);self.serial+=1;
        self.body.push_str(&format!("    let {name}=eval_{k}_{c}(pos);\n"));
        Ok(Expr {text:name,kind:rk})
    }
    fn hoist(&mut self,root:usize,c:usize)->Result<(),String> {
        if crate::envc::var("SDRT_NOHOIST").is_err() {
            for (r,rc) in self.f.gpu_shared(root,c) {self.comp(r,rc)?;}
        }
        Ok(())
    }
    fn invalidate(&mut self,root:usize) {
        for r in self.f.gpu_invalidate(root) {
            self.body.push_str(&format!("    memo_valid[{r}u]=0u; whole_valid[{r}u]=0u;\n"));
        }
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
        let (text, rk) = match i.op {
            0 if i.ty == 1 => (i.ops.first().is_none_or(|&v| v != 0).to_string(), Kind::B),
            0 => {
                let word = literal(c.min(comps(i.ty) - 1))?;
                (format!("bitcast<{}>(0x{word:08x}u)", if tk == Kind::I { "i32" } else { "f32" }), tk)
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
                (cast(&Expr { text: t.into(), kind: Kind::F }, tk), tk)
            }
            2 | 3 => {
                let v = if i.op == 2 { (self.input)(literal(0)?) } else { (self.vp)(arg(0)?) };
                (cast(&component(v, c), tk), tk)
            }
            4 if self.dynamic => {
                let n=comps(i.ty); let base=arg(0)? as u16;
                self.body.push_str(&format!("    if whole_valid[{k}u]==0u {{\n"));
                let mut values=Vec::new();
                for j in 0..n {
                    let idx=self.var_ids[&base.wrapping_add(j as u16)];
                    values.push(pack(&Expr {text:format!("var_{}({idx}u)",typename(tk)),kind:tk}));
                }
                values.resize(4,"0u".into());
                self.body.push_str(&format!("    whole_bits[{k}u]=vec4<u32>({}); whole_valid[{k}u]=1u;\n    }}\n",values.join(",")));
                let e=unpack(format!("whole_bits[{k}u][{c}u]"),tk);(e.text,tk)
            }
            4 => {
                let n = comps(i.ty);
                if !self.reads.contains_key(&k) {
                    // getvar snapshots the whole vector on its first component demand.
                    let base = arg(0)? as u16;
                    let snapshot = (0..n).map(|j| self.vars.get(&base.wrapping_add(j as u16)).cloned().unwrap_or_else(|| lit(0.))).collect();
                    self.reads.insert(k, snapshot);
                }
                let x = self.reads[&k].get(c).cloned().unwrap_or_else(|| lit(0.));
                (cast(&x, tk), tk)
            }
            7 if self.dynamic => {
                let x=self.comp(arg(0)?,c)?;
                let slot=(arg(1)? as u16).wrapping_add(c.min(comps(i.ty)-1) as u16);
                let idx=self.var_ids[&slot];
                self.body.push_str(&format!("    variables[{idx}u]=vec2<u32>({},{}u);\n",pack(&Expr {text:cast(&x,tk),kind:tk}),tk as u32));
                (x.text,x.kind)
            }
            7 => {
                let x = self.comp(arg(0)?, c)?;
                let slot = (arg(1)? as u16).wrapping_add(c.min(comps(i.ty) - 1) as u16);
                self.vars.insert(slot, Expr { text: cast(&x, tk), kind: tk });
                (x.text, x.kind)
            }
            6 => {
                let a = arg(0)?;
                self.whole(a)?;
                let at = self.f.ins.get(a).map_or(4, |i| i.ty);
                let n = comps(at);
                let x = if n > 1 && c >= n {lit(0.)} else if self.dynamic {
                    unpack(format!("whole_bits[{a}u][{}u]",if n==1 {0} else {c}),kind(at))
                } else { self.comp(a, c)? };
                (cast(&x, kind(at)), kind(at))
            }
            9 if self.dynamic => {
                let cond=self.comp(arg(0)?,0)?;
                let at=self.body.len();
                let a=self.comp(arg(1)?,c)?;
                self.hoist(arg(2)?,c)?;
                let yes=self.body.split_off(at);
                self.hoist(arg(1)?,c)?;
                let b=self.comp(arg(2)?,c)?;
                let no=self.body.split_off(at);
                if a.kind!=b.kind {return Err("PP mixed branch types".into());}
                let name=format!("branch_{k}_{c}");
                self.body.push_str(&format!("    var {name}:{};\n    if {} {{\n{yes}    {name}={};\n    }} else {{\n{no}    {name}={};\n    }}\n",typename(a.kind),cast(&cond,Kind::B),a.text,b.text));
                (name,a.kind)
            }
            9 => {
                let cond = self.comp(arg(0)?, 0)?;
                let a = self.comp(arg(1)?, c)?;
                let b = self.comp(arg(2)?, c)?;
                if a.kind != b.kind {
                    return Err("PP mixed branch types".into());
                }
                (format!("select({}, {}, {})", b.text, a.text, cast(&cond, Kind::B)), a.kind)
            }
            11 if self.dynamic => {
                let init=arg(0)?;let cond=arg(1)?;let body=arg(2)?;
                let ty=self.f.ins.get(init).ok_or("PP loop init")?.ty;
                if self.f.ins.get(body).ok_or("PP loop body")?.ty!=ty || i.ty!=ty {return Err("PP mixed loop value types".into());}
                let tail=&i.ops[3..];let start=tail.len()%2;
                let max=*tail.get(start).ok_or("PP loop limit")? as u32 | (*tail.get(start+1).ok_or("PP loop limit")? as u32)<<16;
                self.body.push_str(&format!("    if whole_valid[{k}u]==0u {{\n"));
                self.whole(init)?;
                self.body.push_str(&format!("    var last=whole_bits[{init}u];\n    var iterations=0u;\n    loop {{\n"));
                self.invalidate(cond);self.invalidate(body);
                let condition=self.comp(cond,0)?;
                self.body.push_str(&format!("    if {} {{break;}}\n",cast(&condition,Kind::B)));
                let mut vals=Vec::new();
                for j in 0..comps(ty) {let x=self.comp(body,j)?;vals.push(pack(&Expr {text:cast(&x,kind(ty)),kind:kind(ty)}));}
                vals.resize(4,"0u".into());
                self.body.push_str(&format!("    last=vec4<u32>({});\n    iterations+=1u;\n    if iterations>={}u {{break;}}\n    }}\n    whole_bits[{k}u]=last; whole_valid[{k}u]=1u;\n    }}\n",vals.join(","),if max==0 {1<<24} else {max.min(1<<24)}));
                let e=unpack(format!("whole_bits[{k}u][{c}u]"),tk);(e.text,tk)
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
                (x.text, x.kind)
            }
            13 => {
                let a = arg(0)?;
                let b = arg(1)?;
                let na = self.f.ins.get(a).map_or(0, |i| comps(i.ty));
                let x = if c < na { self.comp(a, c)? } else { self.comp(b, c - na)? };
                (cast(&x, tk), tk)
            }
            15 => {
                let x = self.comp(arg(c)?, 0)?;
                (cast(&x, tk), tk)
            }
            16 => {
                let x = self.comp(arg(0)?, (arg(1)? >> (2 * c)) & 3)?;
                (x.text, x.kind)
            }
            17 => {
                let x = self.comp(arg(0)?, c)?;
                if tk == Kind::I && x.kind == Kind::F {
                    (format!("to_int(floor({}))", x.text), tk)
                } else {
                    (cast(&x, tk), tk)
                }
            }
            18..=22 | 48 | 49 | 54 => {
                let mut vals = Vec::new();
                for r in [arg(0)?, arg(1)?] {
                    let zero = i.op != 20 && c > 0 && comps(i.ty) > 1 && self.f.ins.get(r).is_some_and(|i| comps(i.ty) == 1);
                    let v = if zero {
                        Expr { text: if kind(self.f.ins[r].ty) == Kind::I { "0i" } else { "0.0" }.into(), kind: kind(self.f.ins[r].ty) }
                    } else {
                        self.comp(r, c)?
                    };
                    vals.push(v);
                }
                let rk = if tk == Kind::I || vals.iter().all(|e| e.kind == Kind::I) { Kind::I } else { Kind::F };
                let a = cast(&vals[0], rk);
                let b = cast(&vals[1], rk);
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
                (t, rk)
            }
            23 => {
                let x = self.comp(arg(0)?, c)?;
                let k = if x.kind == Kind::I { Kind::I } else { Kind::F };
                (format!("(-{})", cast(&x, k)), k)
            }
            24 => {
                let a = arg(0)?;
                let b = arg(1)?;
                let ai = self.f.ins.get(a).ok_or("dot operand")?;
                let bi = self.f.ins.get(b).ok_or("dot operand")?;
                let n = comps(ai.ty).min(comps(bi.ty)).max(1);
                let rk = if tk == Kind::I && kind(ai.ty) == Kind::I && kind(bi.ty) == Kind::I { Kind::I } else { Kind::F };
                let mut sum = if rk == Kind::I { "0i" } else { "0.0" }.to_string();
                for j in 0..n {
                    let x = self.comp(a, j)?;
                    let y = self.comp(b, j)?;
                    sum = format!("({sum}+({}*{}))", cast(&x, rk), cast(&y, rk));
                }
                (cast(&Expr { text: sum, kind: rk }, tk), tk)
            }
            26..=34 => {
                let a = self.comp(arg(0)?, 0)?;
                if i.op == 28 {
                    (format!("(!{})", cast(&a, Kind::B)), Kind::B)
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
                    (format!("({} {op} {})", cast(&a, rk), cast(&b, rk)), Kind::B)
                }
            }
            35..=44 | 53 => {
                let a = self.comp(arg(0)?, c)?;
                if i.op == 35 && a.kind == Kind::I {
                    (format!("abs({})", a.text), Kind::I)
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
                    (cast(&Expr { text: format!("{fun}({})", cast(&a, Kind::F)), kind: Kind::F }, tk), tk)
                }
            }
            45 => {
                let a = self.comp(arg(0)?, 0)?;
                let b = self.comp(arg(0)?, 1)?;
                (format!("atan2({},{})", cast(&b, Kind::F), cast(&a, Kind::F)), Kind::F)
            }
            46 => {
                let a = self.comp(arg(0)?, 0)?;
                let b = self.comp(arg(1)?, 0)?;
                (format!("({}*{}({}))", cast(&a, Kind::F), if c == 0 { "cos" } else { "sin" }, cast(&b, Kind::F)), Kind::F)
            }
            47 => {
                let a = self.comp(arg(0)?, c)?;
                let b = self.comp(arg(1)?, c)?;
                let t = self.comp(arg(2)?, c)?;
                let a = cast(&a, Kind::F);
                let b = cast(&b, Kind::F);
                let t = cast(&t, Kind::F);
                (format!("({a}+({b}-{a})*{t})"), Kind::F)
            }
            50 => {
                let a = self.comp(arg(0)?, c)?;
                (format!("({}*random_draw())", cast(&a, Kind::F)), Kind::F)
            }
            51 | 52 if self.dynamic => {
                self.body.push_str(&format!("    if whole_valid[{k}u]==0u {{\n"));
                let x=self.comp(arg(0)?,0)?;let y=self.comp(arg(0)?,1)?;
                let slot=arg(1)?;let filter=i.ops.get(2).copied().unwrap_or(0);
                self.body.push_str(&format!("    whole_bits[{k}u]=bitcast<vec4<u32>>(sample_image({slot}u,vec2<f32>({},{}),{filter}u,{}));\n    whole_valid[{k}u]=1u;\n    }}\n",cast(&x,Kind::F),cast(&y,Kind::F),i.op==52));
                let x=unpack(format!("whole_bits[{k}u][{}u]",if i.op==51 {0} else {c}),Kind::F);(x.text,Kind::F)
            }
            51 | 52 => {
                let sample = if let Some(s) = self.samples.get(&k) {
                    s.clone()
                } else {
                    let x = self.comp(arg(0)?, 0)?;
                    let y = self.comp(arg(0)?, 1)?;
                    let slot = arg(1)?;
                    let filter = i.ops.get(2).copied().unwrap_or(0);
                    let name = format!("sample{k}");
                    self.body.push_str(&format!(
                        "    let {name}=sample_image({slot}u,vec2<f32>({},{}),{filter}u,{});\n",
                        cast(&x, Kind::F),
                        cast(&y, Kind::F),
                        i.op == 52
                    ));
                    self.samples.insert(k, name.clone());
                    name
                };
                (format!("{sample}[{}u]", if i.op == 51 { 0 } else { c }), Kind::F)
            }
            _ => return Err(format!("PP opcode {} not ported", i.op)),
        };
        let name = format!("r{k}_{c}");
        self.body.push_str(&format!(
            "    let {name}:{}={text};\n",
            match rk {
                Kind::F => "f32",
                Kind::I => "i32",
                Kind::B => "bool",
            }
        ));
        let e = Expr { text: name, kind: rk };
        self.memo.insert((k, c), e.clone());
        self.depth -= 1;
        Ok(e)
    }
}

/// Lines of the control-flow shader once the driver inlines every eval_* call: each call site
/// receives a copy of the callee's body and, recursively, of its callees'. Memoization keeps the
/// run time linear, but a register demanded from many sites makes the inlined program grow
/// exponentially with the depth of the expression DAG (chaux node 98: 209 instructions, 98 KB of
/// WGSL, and the NVIDIA compiler passed 12 GB in pipeline creation). Line counts do not depend on
/// literal values, so the preflight and the post-seed compile agree.
fn inlined_lines(functions: &str, body: &str) -> u64 {
    let defs: HashMap<&str, &str> = functions
        .split("fn eval_")
        .skip(1)
        .filter_map(|chunk| chunk.find('(').map(|end| (&chunk[..end], chunk)))
        .collect();
    fn callees(text: &str) -> impl Iterator<Item = &str> {
        text.match_indices("=eval_").filter_map(move |(at, _)| {
            let rest = &text[at + 6..];
            rest.find('(').map(|end| &rest[..end])
        })
    }
    fn cost<'a>(name: &'a str, defs: &HashMap<&'a str, &'a str>, memo: &mut HashMap<&'a str, u64>) -> u64 {
        if let Some(&c) = memo.get(name) {
            return c;
        }
        let Some(text) = defs.get(name).copied() else { return 0 };
        let mut total = text.lines().count() as u64;
        for callee in callees(text) {
            total = total.saturating_add(cost(callee, defs, memo));
        }
        memo.insert(name, total);
        total
    }
    let mut memo = HashMap::new();
    let mut total = body.lines().count() as u64;
    for callee in callees(body) {
        total = total.saturating_add(cost(callee, &defs, &mut memo));
    }
    total
}

pub fn compile(f: &Func, input: &dyn Fn(u32) -> Value, vp: &dyn Fn(usize) -> Value, vars: &[(u16, Value)]) -> Result<String, String> {
    if f.ins.is_empty() || f.ins.len() > 4096 {
        return Err("PP shader size limit".into());
    }
    let dynamic=f.ins.iter().any(|i|i.op==11 && i.ty!=0) || (f.ins.iter().any(|i|i.op==9) && f.ins.iter().any(|i|matches!(i.op,7|50))) || crate::envc::var("SDRT_GPU_DYNAMIC").is_ok();
    if dynamic && f.ins.len()>512 {return Err("PP control-flow shader size limit".into());}
    if f.ins.iter().any(|i|i.op==11 && i.ty!=0) && crate::envc::var("SDRT_LOOPDO").is_ok_and(|v|v!="until") {return Err("GPU supports the default until-loop mode".into());}
    let side=f.gpu_side().to_vec();
    let vars:HashMap<_,_>=vars.iter().map(|&(slot,v)|(slot,component(v,0))).collect();
    let mut slots=std::collections::BTreeSet::new();
    for i in &f.ins {
        if matches!(i.op,4|7) {
            let base=*i.ops.get(if i.op==4 {0} else {1}).ok_or("PP variable slot")?;
            for c in 0..comps(i.ty) {slots.insert(base.wrapping_add(c as u16));}
        }
    }
    let var_ids:HashMap<_,_>=slots.into_iter().enumerate().map(|(idx,slot)|(slot,idx)).collect();
    let mut initial=String::new();
    for (&slot,&idx) in &var_ids {
        if let Some(e)=vars.get(&slot) {initial.push_str(&format!("    variables[{idx}u]=vec2<u32>({},{}u);\n",pack(e),e.kind as u32));}
    }
    // Stable initialization order is also necessary for shader cache reuse.
    let mut lines:Vec<_>=initial.lines().collect();lines.sort();initial=lines.join("\n")+"\n";
    let mut g=Compiler {f,input,vp,vars,reads:HashMap::new(),memo:HashMap::new(),samples:HashMap::new(),side,body:String::new(),depth:0,dynamic,functions:String::new(),serial:0,var_ids};
    let last = f.ins.len() - 1;
    let n = comps(f.ins[last].ty);
    let mut cs = Vec::new();
    for c in 0..n {
        let e = g.comp(last, c)?;
        cs.push(cast(&e, kind(f.ins[last].ty)));
    }
    // Eval::value first packs the declared result type; the pixel output then converts to float.
    let cs: Vec<_> = cs.into_iter().map(|text| cast(&Expr { text, kind: kind(f.ins[last].ty) }, Kind::F)).collect();
    let color = if n >= 4 {
        format!("vec4<f32>({})", cs.join(","))
    } else if n == 3 {
        format!("vec4<f32>({},1.0)", cs.join(","))
    } else {
        format!("vec4<f32>({0},{0},{0},1.0)", cs[0])
    };
    if dynamic {
        // The driver inlines the eval_* calls; past this size its compiler takes minutes and
        // gigabytes (the synthetic control-flow probes stay below 2,000 lines).
        const INLINED_LIMIT: u64 = 100_000;
        let lines = inlined_lines(&g.functions, &g.body);
        if crate::envc::var("SDRT_GPU_LOG").is_ok() {
            eprintln!("GPU PP control flow: {lines} lines once inlined");
        }
        if lines > INLINED_LIMIT {
            return Err(format!("PP control-flow shader too large once inlined ({lines} lines)"));
        }
    }
    let base = include_str!("gpu_prelude.wgsl").split("@compute").next().unwrap();
    let control=if dynamic {format!("var<private> memo_bits:array<vec4<u32>,{0}>;\nvar<private> memo_valid:array<u32,{0}>;\nvar<private> whole_bits:array<vec4<u32>,{0}>;\nvar<private> whole_valid:array<u32,{0}>;\nvar<private> variables:array<vec2<u32>,{1}>;\n{2}\n{3}",f.ins.len(),g.var_ids.len().max(1),include_str!("gpu_control.wgsl"),g.functions)} else {String::new()};
    let initial=if dynamic {initial} else {String::new()};
    Ok(format!("{base}\n{}\n{control}\n@compute @workgroup_size(8,8)\nfn main(@builtin(global_invocation_id) id:vec3<u32>) {{\n    let k=id.y*config.header.z+id.x;\n    if id.x>=config.header.z || k>=config.header.y {{return;}}\n    let pos=(vec2<f32>(id.xy)+vec2<f32>(0.5))/vec2<f32>(f32(config.header.z),f32(config.header.y/config.header.z));\n    rng=pixel_seed(config.header.w,pos);\n{initial}{}    store(k,{color});\n}}\n",include_str!("gpu_pp.wgsl"),g.body))
}
