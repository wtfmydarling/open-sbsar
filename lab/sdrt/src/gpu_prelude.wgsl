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
// A transformation (op 0x04) kernel, gpu_xf.wgsl, is concatenated after this prelude like
// gpu_atomic.wgsl, and gpu_pp.rs prepends it to every Pixel Processor shader, so the
// helpers above and image_pixel below are shared by all GPU kernels.

// Input slot `slot`, pixel k: packed inputs (more than two, options.z) are addressed through
// the descriptors in input_b; otherwise slot 0 is input_a and slot 1 input_b.
fn image_pixel(slot:u32,k:u32)->vec4<f32> {
    if config.options.z!=0u {return input_a[bitcast<vec4<u32>>(input_b[slot*2u+1u]).x+k];}
    if slot==0u {return input_a[k];} return input_b[k];
}

