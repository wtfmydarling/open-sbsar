# open-sbsar draft 1 — function encoding and evaluation

Read [SPEC.md](SPEC.md) for numeric conventions, execution environments and Open IDs. This chapter specifies the active scalar-demand evaluator in `vm.rs::Eval`, not the older unused whole-vector helper.

## 1. Encoding and values

At a function pointer: u16 instruction count, followed by that many variable-length instructions, then zero padding to four-byte alignment. Each instruction has header `op | (type<<6) | (operand_count<<10)` and `operand_count` u16 operands. Bits 0–5 are opcode; 6–9 type; 10–15 operand count. Instruction i defines register i; the last register is the result. Register references are u16. Valid dependency graphs refer backward; `0xffff` can denote an absent operand. Literal u32 words are encoded low halfword first and aligned to four bytes; insert a leading u16 padding operand when necessary and include it in operand count. For literal decoding the CPU skips `operand_count%2` leading halfwords. There is no function byte-length field. [References: `vm.rs::Func::decode`, `literal`; `write.py::Func::bytes`, `function`.]

| Type code | Meaning | Components |
|---|---|---:|
| 0 | string-input UID in encoding; no general string arithmetic | see below |
| 1 | bool | 1 |
| 4…7 | f32, f32x2, f32x3, f32x4 | type−3 |
| 8…11 | i32, i32x2, i32x3, i32x4 | type−7 |
| others | **Open O07** | CPU defaults to one |

The runtime has Void, bool, float vector, integer vector and a string-UID value representation. If the final instruction is input-load opcode 2 with type 0, evaluation returns a string-UID value immediately, without evaluating other registers. The active scalar evaluator otherwise does not implement string expressions: text reads string inputs directly. A scalar read of an absent register returns float zero; whole reads return Void. Missing components of whole vectors read zero. Scalar operands ordinarily broadcast component zero, except the arithmetic zero-extension rule below. Void converts to numeric zero/false. Bool converts to 0 or 1; integer-to-float is numerical f32 conversion; float-to-int saturates/truncates unless opcode 17 explicitly floors. NaN converts to true in a boolean test (`x!=0`). [References: `vm.rs::Value`, `comp_of`, `typed`, `Eval::value`, `comp`.]

## 2. Instruction inventory

`a,b,c` below mean registers unless explicitly called literal. Result typing is as given by the instruction and rules below; no implied eager execution of unused operands.

| Op decimal | Operands | Operation |
|---:|---|---|
| 0 | aligned value words, or bool u16 | constant; bool with no operands is true |
| 1 | literal system-variable index | typed system value |
| 2 | aligned u32 input UID | typed interface input |
| 3 | literal u16 output-slot ID | value previously written by a value processor, default float zero |
| 4 | literal variable-slot ID | snapshot a variable vector |
| 6 | a, optional literal output-slot ID (default 0) | write whole a once, return a |
| 7 | a, literal variable-slot ID | write requested component of a, return it |
| 9 | condition, true branch, false branch | conditional with shared-expression evaluation |
| 11 | initializer, stop condition, body, [pad], u32 limit | pre-tested until loop |
| 12 | a,b | sequence side effects, return b |
| 13 | a,b | concatenate vector components to result width |
| 15 | a,b,c,d | component j = first component of operand j |
| 16 | a, literal packed selector | component j = a[(selector>>(2*j))&3] |
| 17 | a | type cast; float-to-int floors |
| 18,19,20,21,22 | a,b | add, subtract, multiply, divide, modulo |
| 23 | a | negate |
| 24 | a,b | dot |
| 26,27 | a,b | boolean and, or; both operands evaluated |
| 28 | a | boolean not |
| 29,30,31,32,33,34 | a,b | equal, unequal, greater, greater-or-equal, less, less-or-equal |
| 35,36,37 | a | absolute, floor, ceiling |
| 38,39 | a | cosine, sine in radians |
| 40,41,42,43,44 | a | sqrt, natural logarithm, exp, exp2, tan |
| 45 | vector a | atan2(a.y,a.x) |
| 46 | rho,theta | (rho*cos(theta),rho*sin(theta)) |
| 47 | a,b,t | a+(b−a)*t |
| 48,49 | a,b | min,max |
| 50 | a | componentwise a*next_random |
| 51 | position, literal input index, optional literal filter | sample luminance |
| 52 | position, literal input index, optional literal filter | sample RGBA |
| 53 | a | log2 |
| 54 | a,b | power |

Opcodes 5,8,10,14,25 and other unlisted values are **Open O22**; the CPU fallback returns float zero, not evidence of an operation's meaning. For side-effect and volatility propagation and loop-dependency traversal, `reg_operands` treats their first two operands (or fewer if absent) as registers. Their `scalar_reads` fallback is empty: these operands do not become scalar consumers for sharing analysis and are not evaluated by the unknown opcode's compute fallback. Op 3 reads a **write slot**, not a processor ordinal or a node index. For numeric op-2 loads, a UID missing from the runtime numeric-input map supplies a one-component i32 zero before `comp_of` and `typed` conversion. This covers absent binary defaults, pre-v8 image inputs, string inputs read numerically, unknown interface types and absent UIDs unless an override has populated that map; the final string-load shortcut in §1 remains separate. [References: `vm.rs::Eval::compute`, `scalar_binary`, `reg_operands`, `scalar_reads`, `Func::decode`; `ops.rs::vp_slot`, `Runtime::input`; `program.rs::Program::parse`.]

### 2.1 Arithmetic details

For ops 18–22,48,49,54, a scalar operand in a vector result contributes zero to components >0, except multiplication (20), which broadcasts. Evaluate operand a's requested component before b's. Integer arithmetic is selected when the result type is integer **or both operands are integer**. Add/subtract/multiply wrap signed 32-bit. Integer division truncates; division by zero gives zero, and MIN/−1 wraps. Integer modulo is floored: `r=wrapping_rem(p,q); if r!=0 and sign(r)!=sign(q): r+=q`; divisor zero gives zero. Integer power computes `(p as f32).powf(q as f32) as i32` with saturating/truncating conversion; the float-power zero/negative-base guards below do not apply. Thus integer `pow(0,0)=1` and `pow(0,-1)=i32::MAX`. Integer negate/abs at MIN have build-dependent overflow behavior (**Open O02**).

Float divide uses IEEE division. Float modulo is `p-q*floor(p/q)`, preserving separate roundings. In pixel functions divisor zero returns p; in node/value/FX functions it follows the formula and can yield NaN. Float power returns zero for p=0 (including exponent zero or negative) or p<0 with nonintegral q, provided neither operand is NaN; otherwise it uses f32 power. This actual CPU rule supersedes older prose claiming all negative bases return zero.

Dot uses `min(width(a),width(b))`, at least one component, and requests `(a0,b0,a1,b1,…)`. Accumulate products in that order starting at zero. If both operands and result are integer use wrapping integer accumulation; otherwise use f32 accumulation and convert to result type. Comparisons use only component zero: two integers compare exactly; otherwise compare their f32 conversions. NaN comparisons follow IEEE rules. `and` and `or` request both operands in encoded order even when the first determines the result. [Reference: `vm.rs::Eval::compute`, `scalar_binary`.]

`sqrt(x<=0)=0` in every environment. Node/value/FX logarithms return zero when x<=0; pixel log/log2 use f32 mathematical-library behavior. Exp2 is 2^x, not x². All remaining unary operations use the corresponding f32 operation, except CPU-context sine/cosine below. `toint` floors float input before the saturating cast; ordinary conversions elsewhere truncate. [Reference: `vm.rs::Eval::compute`.]

### 2.2 CPU-context sine and cosine

Node, value and FX parameter functions use the following f32 polynomial; pixel functions use f32 sin/cos. Angles in the VM are radians. Define:

```
trig_turns(u,phase):
    u=u+phase
    u=abs(u%1)
    if u>.5: u=1-u
    x=u*4-1
    x2=x*x; x3=x*x2; x5=x3*x2; x7=x5*x2
    return x*f32bits(0x3fc90fdb)
         + x3*f32bits(0xbf255de7)
         + x5*f32bits(0x3da335e3)
         + x7*f32bits(0xbb94463d)
sin(t)=trig_turns(t*(1/TAU_f32),.25)
cos(t)=trig_turns(t*(1/TAU_f32),.5)
```

Keep the left-associated sum and intermediate f32 products. Cartesian uses this same sine/cosine in CPU contexts. Tan and atan2 retain library behavior. FX stamp raster rotation is a different calculation and does not use this polynomial. [Reference: `vm.rs::sbs_trig_turns`, `sbs_sin`, `sbs_cos`, `Eval::compute`.]

## 3. Lazy evaluation and sharing

Maintain a memo entry per `(register,component)` and a separate whole-value cache for operations with whole-value effects. Evaluate result components from zero upward. On demand, scalar registers map any requested component to zero; vector component requests clamp at three. Return cached components without effects or random draws. Otherwise evaluate required operands in order, compute, cache and return. A random component draws only when first demanded.

Variables are scalar slots. Setting a vector at slot k writes component c to k+c. Get-variable snapshots **all declared components** on its first component demand. Later writes do not alter that register until a loop invalidates it. Unset slots read zero, including holes in a partially initialized vector. A set-variable instruction writes only its demanded component unless a whole-value context demands all components. Variables are shared across size, seed and parameter functions of a node, across an FX traversal, and through a value processor. Every pixel processor invocation gets its own copy of the initial node variables. [References: `vm.rs::var_read`, `set_var_comp`, `compute`; `ops.rs::pixel_processor`; `fxmap.rs::Ctx::eval_mode`.]

Mark an instruction side-effecting if its opcode is 6,7 or 11, or any of its register operands is side-effecting. Sequence first executes the whole first operand if side-effecting, then the whole second operand if side-effecting, each once; finally returns the requested component of the second. Pure first operands are not executed solely for sequencing. Random by itself is not marked a side effect for this rule. A write evaluates its whole source once and calls the environment's write hook once per cache lifetime; value-processor publication is deferred as specified in §4. [Reference: `vm.rs::Func::decode`, `Eval::compute` ops 6,12.]

### 3.1 Component liveness and shared expressions

To reproduce conditional random order, count distinct **scalar consumers**, not register consumers. Start by marking all final-result components needed. Walk registers backward. For each needed component `(k,c)`, enumerate its scalar reads using the table below, keep only earlier valid registers, map scalar operands' components to zero, and mark each read needed. Deduplicate identical reads within that consumer. A read with two or more distinct consumer pairs `(k,c)` is shared.

| Instruction | Scalar reads for output component c, in order |
|---|---|
| 0…4 | none |
| 6 | all components of a |
| 7,17,23,35…44,50,53 | a.c |
| 9 | condition.0, true.c, false.c |
| 11 | all initializer, condition.0, all body |
| 12 | all a (even when pure); all b if side-effecting; b.c |
| 13 | a.c if c<width(a), otherwise b.(c−width(a)) |
| 15 | operand[c].0 |
| 16 | a.selected_component |
| 18…22,48,49,54 | a.c then b.c, omitting zero-extended scalar tails |
| 24 | a.j,b.j for j ascending through dot width |
| 26,27,29…34,46 | a.0,b.0 |
| 28 | a.0 |
| 45,51,52 | a.0,a.1 |
| 47 | a.c,b.c,t.c |

A conditional first computes the condition. Visit its true branch, then false branch, regardless of the choice. For the chosen branch demand its component normally. For the unchosen branch traverse the above read graph depth-first in operand order. At a shared component, stop descending: evaluate it unless its own opcode is write/set-variable/loop (6/7/11), in which case skip it and its descendants. At a nonshared component continue descending without executing it. Deduplicate found shared components in discovery order. This can consume random values in an unchosen branch, but only through live shared expressions. [References: `vm.rs::scalar_reads`, `Func::decode`, `Eval::compute` op 9.]

### 3.2 Loops

Op 11 operands start with initializer, stop-condition and body registers, followed by an aligned u32 maximum. Decode padding in this tail as for a literal. A zero maximum is unlimited except the CPU's safety cap of 2^24 body executions (**Open O08** as a format limit).

```
last=whole(initializer)
iterations=0
repeat:
    invalidate(condition)
    invalidate(body)
    if component(condition,0) is true: break
    last=whole_components_in_order(body)
    iterations+=1
    if (maximum!=0 and iterations>=maximum) or iterations>=2^24: break
cache last without conversion to the loop instruction type
return comp_of(last, requested_component)
```

This is an **until** loop, not do-while. No body executes if the initial stop condition is true. Cache the loop's whole result so later component demands do not rerun it. The cached value retains the last body's type/width (or the initializer's when no body runs); op 11 returns its components through `comp_of` without `typed`. This does not remove all declared-type effects: `comp` maps requested components according to the loop instruction's width, and `value(loop)` assembles/converts components into that declared type and width, including when the loop is the final function result. `whole_of(loop)` returns the native cached value if already present; otherwise it calls `value(loop)` and caches that assembled value. [References: `vm.rs::Eval::compute` op 11, `comp`, `value`, `whole_of`, `comp_of`.]

Volatile registers are variable reads/writes, output writes, loops and random (4,6,7,11,50), plus instructions depending on them. For each condition/body root, invalidate the root and recursively only volatile dependencies; keep invariant constants and pure subgraphs memoized. Invalidating a register clears both scalar and whole caches. [References: `vm.rs::Func::decode`, `Eval::invalidate`.]

## 4. Environments and samplers

System variables: index 1 is float2 `$size`; 3 is float2 `$sizelog2` computed by log2 of size; 8 is float2 `$pos`; 10 is scalar `$number`; 11 is `$depthpow2`; 12 is `$depth`. Node context gives .5,.5 position and number zero. Pixel context gives pixel centres. FX context gives traversal centre, current iteration number, 2^depth and depth. Other variables, including time in the current runtime, return scalar zero; broader time/system-variable behavior is **Open O22**. [References: `ops.rs::size_sysvar`, `NodeEnv::sysvar`, `PixEnv::sysvar`; `fxmap.rs::QuadEnv::sysvar`.]

Sampler position is computed x then y, and the entire sample is cached when any output component is demanded. Filter defaults to zero; bit 0 means bilinear, otherwise nearest; bit 1 clamps both axes, otherwise both wrap. Pixel sampling ignores the processor node's own tiling byte. Sample the selected input at its native dimensions. Missing input indices return transparent black. Seed, size and ordinary node-parameter functions have empty image-source lists, so ops 51/52 always sample transparent black there, regardless of filter. Pixel bilinear uses coupled texture sampling; value-processor and FX function bilinear uses exact float weights. For samplelum from a colour image use `(R+G+B)/3`, not alpha or weighted luminance; gray inputs use R. Samplecol returns RGBA, with gray images already replicated. [References: `vm.rs::Eval::compute` ops 51/52; `ops.rs::NodeEnv::sample`, `PixEnv::sample`, `seed`, `slot`, `run_inner`; `fxmap.rs::QuadEnv::sample`.]

Value-processor writes append `(output-slot ID, whole source value)` to a pending list. Only after the entire function returns are these entries inserted into the persistent slot map, in execution order; the last write to a slot wins. Op 3 cannot see earlier writes from the same processor evaluation: it reads the value committed by an earlier node, or scalar float zero for an absent slot. The stored source value is not converted to op 6's declared type; op 3 applies `typed` to each component it reads. [References: `ops.rs::NodeEnv::write`, `vp_slot`, `pixel_processor`; `vm.rs::Eval::compute` ops 3/6.]

Pixel output is the final function value: float4 uses RGBA; width 3 supplies alpha one; width 1 or 2 replicates the first component into RGB and supplies alpha one. Gray uses only the first component. The active processor implementation does not substitute its recorded write side effect for the final return value. [Reference: `ops.rs::pixel_processor`.]
