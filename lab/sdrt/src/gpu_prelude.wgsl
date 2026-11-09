// Independent kernels matching the CPU reference. Storage uses RGBA f32;
// every node quantizes on the GPU before its consumers run.
struct Config {
    header: vec4<u32>, // operation, pixel count, width, alpha flatten
    values: vec4<f32>, // uniform color or grayscale weights
    extra: vec4<f32>,  // matte
    selectors: vec4<u32>,
    options: vec4<u32>, // format, color, packed inputs, input count
    parameters: array<vec4<f32>,5>, // node-specific constants
}
@group(0) @binding(0) var<storage, read> config: Config;
@group(0) @binding(1) var<storage, read> input_a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> input_b: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<vec4<f32>>;
// q/(2^bits-1) is the repeating binary fraction .qqqq... . Construct
// its correctly rounded f32 representation instead of an approximate GPU reciprocal.
fn unorm_float(q:u32,bits:u32)->f32 {
    if q==0u { return 0.0; }
    if q==((1u<<bits)-1u) { return 1.0; }
    let k=firstLeadingBit(q);
    var mant:u32;
    if bits==8u {
        let repeat=q*0x01010101u;
        mant=(repeat>>(k+1u))+((repeat>>k)&1u);
    } else {
        let repeat=(q<<16u)|q;
        if k<7u { mant=repeat<<(7u-k); }
        else {
            mant=repeat>>(k-7u);
            if k>=8u { mant+=(repeat>>(k-8u))&1u; }
        }
    }
    return bitcast<f32>(((k+127u-bits)<<23u)|(mant&0x7fffffu));
}
fn quantize(v: f32) -> f32 {
    let format=config.options.x;
    if format < 2u {
        // Rust's float-to-integer cast maps NaN to zero.
        var x=clamp(v,0.0,1.0);
        if (bitcast<u32>(v)&0x7fffffffu)>0x7f800000u { x=0.0; }
        if format==0u {
            let i=u32(floor(x*4096.0));
            return unorm_float((i*255u+2047u)>>12u,8u);
        }
        let i=u32(floor(x*1048576.0));
        // Exact 36-bit product reduced algebraically, no shader int64 required.
        let correction=i32((i&15u)*65536u)-i32(i)+524287;
        var q=i32(i>>4u);
        if correction<0 { q-=1; } else if correction>=1048576 { q+=1; }
        return unorm_float(u32(q),16u);
    }
    if format==2u {
        let bits=bitcast<u32>(v); let sign=bits&0x80000000u;
        let exp=i32((bits>>23u)&255u)-127;
        if exp==128 { return v; }
        if exp>15 { return bitcast<f32>(sign|0x477fe000u); }
        if exp< -24 { return bitcast<f32>(sign); }
        let keep=min(10,24+exp);
        let shift=u32(23-keep); let tail=bits&((1u<<shift)-1u);
        var r=bits-tail; let mid=1u<<(shift-1u);
        if tail>mid || (tail==mid && ((r>>shift)&1u)!=0u) { r+=1u<<shift; }
        return bitcast<f32>(r);
    }
    return v;
}
fn store(k:u32, p:vec4<f32>) {
    if config.options.y==0u {
        let g=quantize(p.x); output[k]=vec4<f32>(g,g,g,1.0);
    } else { output[k]=vec4<f32>(quantize(p.x),quantize(p.y),quantize(p.z),quantize(p.w)); }
}
// A product's correctly rounded value, opaque to the shader compiler. The CPU rounds every
// multiply and add on its own; the driver may contract a multiply into the add that follows
// it (one rounding instead of two: a different last bit). The XOR operand is zero at run time
// (the pixel count is below 2^31) but unknown to the compiler, so the multiply is rounded and
// the addition happens on the rounded value. Wrap each product that feeds an addition.
fn barrier(p:f32)->f32 {
    return bitcast<f32>(bitcast<u32>(p)^(config.header.y>>31u));
}
fn barrier4(p:vec4<f32>)->vec4<f32> {
    return bitcast<vec4<f32>>(bitcast<vec4<u32>>(p)^vec4<u32>(config.header.y>>31u));
}

// --- Rust's f32 operations and correctly rounded arithmetic ---
// Besides contracting a*b + c into an fma, the driver distributes a constant factor over a sum
// with a constant: 6.0 * (d + E) became fma(6, d, 6E), one ULP off. The kernels therefore wrap in
// barrier() every product that feeds an addition and every sum that feeds a product, as well as
// the constant factor itself.
//
// The GPU's division and square root are not correctly rounded (WGSL allows 2.5 ULP), while the
// CPU (the specification) rounds every operation on its own. div_rn and sqrt_rn below reproduce
// IEEE round to nearest even for finite operands.

// The magnitude of `m` with the sign bit of `s`.
fn sign_of(m: f32, s: f32) -> f32 {
    return bitcast<f32>((bitcast<u32>(m) & 0x7fffffffu) | (bitcast<u32>(s) & 0x80000000u));
}

// Whether `v` is a NaN (any payload).
fn nan_bits(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;
}

// Whether `v` is finite.
fn finite_bits(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

// Rust's f32::round: half away from zero, the sign of zero kept. Exact for every finite value
// (floor(v + 0.5) is not: v + 0.5 rounds, e.g. 0.49999997 + 0.5 is 1.0).
fn rs_round(v: f32) -> f32 {
    let a = abs(v);
    let fl = floor(a);
    var r = fl;
    if a - fl >= 0.5 { r = fl + 1.0; }
    return sign_of(r, v);
}

// Rust's f32::clamp: NaN stays NaN, -0.0 stays -0.0.
fn rs_clamp(v: f32, lo: f32, hi: f32) -> f32 {
    var x = v;
    if x < lo { x = lo; }
    if x > hi { x = hi; }
    return x;
}

// Rust's f32::max / f32::min: a NaN operand yields the other one; equal values keep the
// second operand (measured on Rust: max(0.0, -0.0) = -0.0 and min(0.0, -0.0) = -0.0).
fn rs_max(a: f32, b: f32) -> f32 {
    if nan_bits(a) { return b; }
    if nan_bits(b) { return a; }
    return select(b, a, a > b);
}

fn rs_min(a: f32, b: f32) -> f32 {
    if nan_bits(a) { return b; }
    if nan_bits(b) { return a; }
    return select(b, a, a < b);
}

// floor(v + 0.5) computed exactly (the CPU adds 0.5 in f64), v >= 0.
fn floor_half_up(v: f32) -> f32 {
    let fl = floor(v);
    if v - fl >= 0.5 { return fl + 1.0; }
    return fl;
}

// Rust's x % 1.0 (fmod, exact): the fraction with the sign of x. WGSL's % is x - trunc(x/y)*y,
// rounded; x - trunc(x) is exact for every finite x.
fn rs_fmod1(x: f32) -> f32 {
    return sign_of(x - trunc(x), x);
}

// Rust's x.rem_euclid(1.0): x % 1.0, plus 1.0 (rounded) when negative.
fn rs_rem1(x: f32) -> f32 {
    let r = rs_fmod1(x);
    if r < 0.0 { return r + 1.0; }
    return r;
}

// 2^k for -126 <= k <= 127.
fn pow2i(k: i32) -> f32 {
    return bitcast<f32>(u32((k + 127) << 23));
}

// The exact product of two u32, as (high, low).
fn mul_u32(a:u32,b:u32)->vec2<u32> {
    let a0=a&0xffffu;let a1=a>>16u;
    let b0=b&0xffffu;let b1=b>>16u;
    let t0=a0*b0;
    let t1=a0*b1;
    let t2=a1*b0;
    let t3=a1*b1;
    // the middle column: (t0 >> 16) + (t1 & 0xffff) + (t2 & 0xffff), which needs 18 bits
    let mid=(t0>>16u)+(t1&0xffffu)+(t2&0xffffu);
    let carry=mid>>16u;
    let lo=((mid&0xffffu)<<16u)|(t0&0xffffu);
    let hi=t3+(t1>>16u)+(t2>>16u)+carry;
    return vec2<u32>(hi,lo);
}

// The exact sum of two u64, as (high, low).
fn add_u64(a:vec2<u32>,b:vec2<u32>)->vec2<u32> {
    let lo=a.y+b.y;
    let carry=select(0u,1u,lo<a.y);
    return vec2<u32>(a.x+b.x+carry,lo);
}

// Whether the unsigned 64-bit value a is at least b.
fn ge_u64(a:vec2<u32>,b:vec2<u32>)->bool {
    if a.x!=b.x {return a.x>b.x;}
    return a.y>=b.y;
}

// sqrt(x) correctly rounded (IEEE round to nearest even), as Rust's f32::sqrt. The GPU's square
// root is not correctly rounded (WGSL allows 2.5 ULP), so the root is found exactly instead.
//
// The radicand is split as x = m * 2^E with m in [1, 2) (a subnormal radicand is normalised
// first) and scaled to r = m * 2^(E & 1), so that r lies in [1, 4) and its root in [1, 2): the
// scaling is by a power of two and therefore exact.
//
// The root is then found on the uniform grid S * 2^-23 (S an integer in [2^23, 2^24]) by an exact
// integer binary search. With the radicand in units of 2^-46 (rad = r * 2^23, an integer) the
// root is below the midpoint of s and s + 2^-23 exactly when
//     4*rad <= 4*S*S + 4*S + 1
// (the midpoint is s + 2^-24, whose square is S^2 + S/2 + 1/4 in those units; the comparison is
// exact because rad is an integer). The predicate is monotone in S, so the smallest S that
// satisfies it is the correctly rounded root: no tie is possible, since the square root of a
// float is never exactly halfway between two floats.
//
// The integer arithmetic is exact, so the result is the correctly rounded root for every finite
// radicand (a subnormal result is flushed to zero by the driver, like the driver's own square
// root; the CPU keeps it).
fn sqrt_rn(x: f32) -> f32 {
    let bits=bitcast<u32>(x);
    let sign=bits&0x80000000u;
    let exp=(bits>>23u)&255u;
    let mant=bits&0x7fffffu;
    if exp==255u {
        if mant!=0u {return x;}                        // a NaN: the payload is kept
        if sign!=0u {return bitcast<f32>(0xffc00000u);}   // -infinity: a NaN
        return x;                                      // +infinity
    }
    if sign!=0u {
        if exp==0u && mant==0u {return x;}             // -0.0
        return bitcast<f32>(0xffc00000u);              // a negative radicand: a NaN
    }
    if exp==0u && mant==0u {return x;}                  // +0.0
    // x = m * 2^e with m = M * 2^-23 in [1, 2) and e the exponent.
    var m:u32;
    var e:i32;
    if exp==0u {
        // a subnormal radicand: mant * 2^-149 = m * 2^e
        let l=i32(firstLeadingBit(mant));
        m=((mant-(1u<<u32(l)))<<u32(23-l))|0x800000u;
        e=l-149;
    } else {
        m=mant|0x800000u;
        e=i32(exp)-127;
    }
    // r = m * 2^(e & 1) in [1, 4): its integer in units of 2^-23 is rp.
    let odd=e&1;
    let rp=m<<u32(odd);
    // The radicand in units of 2^-46, times 4: rad4 = 4 * r * 2^23.
    let rad4=mul_u32(rp,33554432u);
    // The search for the smallest S in [2^23, 2^24] with rad4 <= 4*S*S + 4*S + 1, that is
    // 4*rad <= 4*S*S + 4*S + 1. The predicate holds for S = 2^24 (the radicand is below 4).
    var lo=0x800000u;
    var hi=0x1000000u;
    for (var i=0u;i<25u;i++) {
        if lo>=hi {break;}
        let mid=(lo+hi)>>1u;
        let g=add_u64(mul_u32(mid<<2u,mid),vec2<u32>(0u,(mid<<2u)+1u));
        if ge_u64(g,rad4) {hi=mid;} else {lo=mid+1u;}
    }
    // The float with the integer lo in units of 2^-23 (lo - 2^23 is its mantissa).
    let s=bitcast<f32>(0x3f800000u+(lo-0x800000u));
    // x = r * 2^(e - odd): the root is s * 2^((e - odd) / 2), an exact scaling.
    return s*pow2i((e-odd)/2);
}

// The exact difference a - b, as (high, low); the borrow is taken from the low word.
fn sub_u64(a:vec2<u32>,b:vec2<u32>)->vec2<u32> {
    let lo=a.y-b.y;
    let borrow=select(0u,1u,lo>a.y);
    let hi=a.x-b.x-borrow;
    return vec2<u32>(hi,lo);
}

// The quotient q = floor(N/D) and the remainder r = N - q*D, for N below 2^48 and D in
// [2^23, 2^24): the quotient is below 2^25. The binary search is exact.
fn div_qr(N:vec2<u32>,D:u32)->vec2<u32> {
    var lo=0u;
    var hi=0x02000000u;
    while (lo<hi) {
        let mid=(lo+hi+1u)>>1u;
        if (ge_u64(mul_u32(mid,D),N)) {hi=mid-1u;} else {lo=mid;}
    }
    let q=lo;
    let r=sub_u64(N,mul_u32(q,D)).y;
    return vec2<u32>(q,r);
}

// a / b correctly rounded (IEEE round to nearest even), as the CPU divides for finite operands.
// The GPU's division is approximate (WGSL allows 2.5 ULP), so the quotient is computed in exact
// integer arithmetic instead:
//
// Both operands are normalised to a = A * 2^(ea-23) and b = B * 2^(eb-23), with A and B in
// [2^23, 2^24) (a subnormal operand is normalised first), so that the quotient is A/B * 2^s with
// s = ea - eb. The result's exponent E follows from A >= B or A < B, and the exact quotient is
// then the integer division N/B with N = A * 2^(23 - E + s), rounded to nearest even. The
// rounding is decided on the exact remainder, so the result is the correctly rounded quotient for
// every finite pair of operands (including a subnormal result, which the driver would flush to
// zero in its own arithmetic).
fn div_rn(a: f32, b: f32) -> f32 {
    let ba=bitcast<u32>(a);
    let bb=bitcast<u32>(b);
    let sign=(ba^bb)&0x80000000u;
    // A non-finite operand: the driver's own division (IEEE, apart from the NaN payload).
    if ((ba&0x7f800000u)==0x7f800000u) || ((bb&0x7f800000u)==0x7f800000u) {return a/b;}
    if (ba&0x7fffffffu)==0u {return bitcast<f32>(sign);}
    if (bb&0x7fffffffu)==0u {return bitcast<f32>(sign|0x7f800000u);}
    // The normalised mantissas and exponents.
    var A=(ba&0x007fffffu)|0x800000u;
    var ea=i32((ba>>23u)&255u)-127;
    var B=(bb&0x007fffffu)|0x800000u;
    var eb=i32((bb>>23u)&255u)-127;
    if ((ba&0x7f800000u)==0u) {
        let l=i32(firstLeadingBit(ba&0x007fffffu));
        A=(ba&0x007fffffu)<<u32(23-l);
        ea=l-149;
    }
    if ((bb&0x7f800000u)==0u) {
        let l=i32(firstLeadingBit(bb&0x007fffffu));
        B=(bb&0x007fffffu)<<u32(23-l);
        eb=l-149;
    }
    let s=ea-eb;
    // The result's exponent and the numerator of the exact quotient.
    var E=s;
    var shift=23;
    if (A<B) {E=s-1;shift=24;}
    var sub=false;
    if (E<-126) {sub=true;E=-126;shift=s+149;}
    if (shift<0) {
        // A result below 2^-149: the rounded mantissa is 0 or 1 (the value is at least 2^-150
        // exactly when A * 2^(s+150) >= B).
        var m=0u;
        let k=-shift-1;
        if (k<25 && ge_u64(vec2<u32>(0u,A),mul_u32(B,1u<<u32(k)))) {m=1u;}
        return bitcast<f32>(sign|m);
    }
    let qr=div_qr(mul_u32(A,1u<<u32(shift)),B);
    var q=qr.x;
    let r=qr.y;
    // Round to nearest, ties to even.
    if (r+r>B) {q=q+1u;} else if (r+r==B) {if (q&1u)!=0u {q=q+1u;}}
    if (sub) {
        // The subnormal grid: the value is q * 2^-149 (q below 2^23; q = 2^23 is 2^-126).
        return bitcast<f32>(sign|q);
    }
    // The normal grid: the value is (q - 2^23) * 2^(E-23) with the exponent field E + 127.
    var e=E;
    var mant=q;
    if (mant>=0x1000000u) {mant=0x800000u;e=e+1;}
    if (e>127) {return bitcast<f32>(sign|0x7f800000u);}
    return bitcast<f32>(sign|(u32(e+127)<<23u)|(mant-0x800000u));
}

// The 16-bit code of a UNORM texel, (v * 65535).round() as u32 (Rust's cast saturates).
fn unorm16_code(v: f32) -> u32 {
    if nan_bits(v) { return 0u; }
    let r = rs_round(barrier(v * 65535.0));
    if r <= 0.0 { return 0u; }
    if r >= 4294967295.0 { return 4294967295u; }
    return u32(r);
}
// A transformation (op 0x04) kernel, gpu_xf.wgsl, is concatenated after this prelude like
// gpu_atomic.wgsl, and gpu_pp.rs prepends it to every Pixel Processor shader, so the
// helpers above and image_pixel below are shared by all GPU kernels.

// Input slot `slot`, pixel k: packed inputs (more than two, options.z) are addressed through
// the descriptors in input_b; otherwise slot 0 is input_a and slot 1 input_b.
fn image_pixel(slot:u32,k:u32)->vec4<f32> {
    if config.options.z!=0u {return input_a[bitcast<vec4<u32>>(input_b[slot*2u+1u]).x+k];}
    if slot==0u {return input_a[k];} return input_b[k];
}

