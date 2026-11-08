var<private> rng:u32;
fn pixel_seed(seed:u32,pos:vec2<f32>)->u32 {
    let words=vec3<u32>(seed,bitcast<u32>(pos.x),bitcast<u32>(pos.y));
    var h=0u;
    for(var j=0u;j<3u;j++) {h+=words[j];h+=h<<10u;h^=h>>6u;}
    h+=h<<3u;h^=h>>11u;h+=h<<15u;return h&0xffffffu;
}
fn random_draw()->f32 {
    rng=(3626811u*rng+51137u)&0xffffffu;
    return f32(rng>>1u)*(1.0/8388608.0);
}
fn to_int(x:f32)->i32 {
    if (bitcast<u32>(x)&0x7fffffffu)>0x7f800000u {return 0i;}
    if x>=2147483648.0 {return 2147483647i;}
    if x<= -2147483648.0 {return bitcast<i32>(0x80000000u);}
    return i32(x);
}
fn int_div(a:i32,b:i32)->i32 {
    if b==0i {return 0i;}
    if a==bitcast<i32>(0x80000000u) && b== -1i {return a;}
    return a/b;
}
fn int_mod(a:i32,b:i32)->i32 {
    if b==0i || b== -1i {return 0i;}
    // Unsigned magnitudes avoid backend-dependent signed remainder lowering.
    let ua=bitcast<u32>(a); let ub=bitcast<u32>(b);
    let mag=select(ua,0u-ua,a<0i)%select(ub,0u-ub,b<0i);
    let r=select(i32(mag),-i32(mag),a<0i);
    if r!=0i && ((r<0i)!=(b<0i)) {return r+b;}
    return r;
}
fn float_mod(a:f32,b:f32)->f32 {
    if b==0.0 {return a;}
    return a-b*floor(a/b);
}
fn pp_sqrt(x:f32)->f32 {if x<=0.0 {return 0.0;} return sqrt(x);}
fn pp_pow(a:f32,b:f32)->f32 {
    let an=(bitcast<u32>(a)&0x7fffffffu)>0x7f800000u;
    let bn=(bitcast<u32>(b)&0x7fffffffu)>0x7f800000u;
    if !an && !bn && (a==0.0 || (a<0.0 && b!=floor(b))) {return 0.0;}
    // Match the CPU reference's powf exceptions before propagating NaN.
    if b==0.0 || a==1.0 {return 1.0;}
    if an || bn {return bitcast<f32>(0x7fc00000u);}
    let infinite_b=(bitcast<u32>(b)&0x7fffffffu)==0x7f800000u;
    if infinite_b && abs(a)==1.0 {return 1.0;}
    let p=pow(abs(a),b);
    if !infinite_b && a<0.0 && (b-2.0*floor(b/2.0))!=0.0 {return -p;}
    return p;
}
fn int_pow(a:i32,b:i32)->i32 {
    if a==0i {if b==0i {return 1i;} if b<0i {return 2147483647i;} return 0i;}
    return to_int(pp_pow(f32(a),f32(b)));
}
// CPU fetch converts floor coordinates to saturating i64, then tiles/clamps. Node dimensions
// are powers of two, so even large f32 coordinates can be wrapped exactly without shader i64.
fn address(x:f32,n:u32,clamped:bool)->u32 {
    let bits=bitcast<u32>(x);let ab=bits&0x7fffffffu;
    if ab>0x7f800000u {return 0u;}
    if clamped {
        if x<=0.0 {return 0u;} if x>=f32(n-1u) {return n-1u;} return u32(x);
    }
    if x>=9223372036854775808.0 {return n-1u;}
    if x<= -9223372036854775808.0 {return 0u;}
    if x>= -2147483648.0 && x<2147483648.0 {return u32(int_mod(i32(x),i32(n)));}
    let shift=(ab>>23u)-127u-23u;
    if shift>=firstLeadingBit(n) {return 0u;}
    let v=((ab&0x7fffffu|0x800000u)<<shift)&(n-1u);
    return select(v,(0u-v)&(n-1u),x<0.0);
}
fn texel(slot:u32,xy:vec2<f32>,sz:vec2<u32>,clamped:bool)->vec4<f32> {
    let k=address(xy.y,sz.y,clamped)*sz.x+address(xy.x,sz.x,clamped);
    return image_pixel(slot,k);
}
fn nearest_coord(u:f32,n:u32)->f32 {
    let b=bitcast<u32>(u); let a=b&0x7fffffffu;
    if a>0x7f800000u {return 0.0;}
    if u>=9223372036854775808.0/f32(n) {return 9223372036854775808.0;}
    if u<= -9223372036854775808.0/f32(n) {return -9223372036854775808.0;}
    return floor(u*f32(n));
}
fn sample_image(slot:u32,uv:vec2<f32>,sample_mode:u32,color:bool)->vec4<f32> {
    if slot>=config.options.w {return vec4<f32>(0.0);}
    var desc:vec4<u32>;
    if config.options.z!=0u {desc=bitcast<vec4<u32>>(input_b[slot*2u]);}
    else {desc=select(bitcast<vec4<u32>>(config.values),bitcast<vec4<u32>>(config.extra),slot==1u);}
    if desc.x==0u || desc.y==0u {return vec4<f32>(0.0);}
    let clamp_edges=(sample_mode&2u)!=0u;
    var p:vec4<f32>;
    if (sample_mode&1u)==0u {
        p=texel(slot,vec2<f32>(nearest_coord(uv.x,desc.x),nearest_coord(uv.y,desc.y)),desc.xy,clamp_edges);
    } else {
        if (bitcast<u32>(uv.x)&0x7fffffffu)>=0x7f800000u || (bitcast<u32>(uv.y)&0x7fffffffu)>=0x7f800000u ||
           abs(uv.x)>bitcast<f32>(0x7f7fffffu)/f32(desc.x) || abs(uv.y)>bitcast<f32>(0x7f7fffffu)/f32(desc.y) {
            // CPU's non-finite weights cast to zero for UNORM, propagate for float.
            if desc.z<=1u {return vec4<f32>(0.0);}
            return vec4<f32>(bitcast<f32>(0x7fc00000u));
        }
        let xy=uv*vec2<f32>(desc.xy)-vec2<f32>(0.5);let base=floor(xy);
        // Positive fractional parts, round ties up (WGSL round would use ties-to-even).
        let f=vec2<u32>(floor((xy-base)*256.0+vec2<f32>(0.5)));
        let cross=(f.x*f.y+128u)>>8u;
        let weights=vec4<u32>(256u-f.x-f.y+cross,f.x-cross,f.y-cross,cross);
        let a=texel(slot,base,desc.xy,clamp_edges);
        let b=texel(slot,base+vec2<f32>(1.0,0.0),desc.xy,clamp_edges);
        let c=texel(slot,base+vec2<f32>(0.0,1.0),desc.xy,clamp_edges);
        let d=texel(slot,base+vec2<f32>(1.0,1.0),desc.xy,clamp_edges);
        if desc.z<=1u {
            let sum=vec4<u32>(floor(a*65535.0+vec4<f32>(0.5)))*weights.x+
                    vec4<u32>(floor(b*65535.0+vec4<f32>(0.5)))*weights.y+
                    vec4<u32>(floor(c*65535.0+vec4<f32>(0.5)))*weights.z+
                    vec4<u32>(floor(d*65535.0+vec4<f32>(0.5)))*weights.w;
            let q=(sum+vec4<u32>(128u))>>vec4<u32>(8u);
            p=vec4<f32>(unorm_float(q.x,16u),unorm_float(q.y,16u),unorm_float(q.z,16u),unorm_float(q.w,16u));
        } else {
            let wt=vec4<f32>(weights)*(1.0/256.0);
            p=((a*wt.x+b*wt.y)+c*wt.z)+d*wt.w;
        }
    }
    if !color && desc.w!=0u {let g=((p.x+p.y)+p.z)/3.0;return vec4<f32>(g,g,g,1.0);}
    return p;
}
