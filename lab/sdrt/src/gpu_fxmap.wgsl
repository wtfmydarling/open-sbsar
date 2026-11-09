// Exact square of a 24-bit integer, represented as low/high u32 limbs.
fn fx_square(m:u32)->vec2<u32> {
    let a=m&65535u; let b=m>>16u;
    let ll=a*a; let mid=2u*a*b;
    let lo=ll+(mid<<16u);
    return vec2<u32>(lo,b*b+(mid>>16u)+select(0u,1u,lo<ll));
}
// Correctly rounded sqrt of a normal nonnegative f32. Native sqrt only seeds an
// integer search. N is the exact 47/48-bit radicand; rounding compares it with
// m*m+m (the square of m+1/2 is a quarter larger, so an integer N cannot tie).
fn fx_sqrt(x:f32)->f32 {
    let bits=bitcast<u32>(x); let exp=i32((bits>>23u)&255u)-127;
    if x==0.0 || exp == -127 { return 0.0; }
    let odd=exp&1;
    let mant=(bits&0x7fffffu)|0x800000u;
    let n=u64_shl32(mant,u32(23+odd));
    let norm=bitcast<f32>((u32(127+odd)<<23u)|(bits&0x7fffffu));
    var m=u32(sqrt(norm)*8388608.0);
    loop { if u64_ge(n,fx_square(m)) {break;} m-=1u; }
    loop { if !u64_ge(n,fx_square(m+1u)) {break;} m+=1u; }
    let threshold=u64_add(fx_square(m),vec2<u32>(m,0u));
    if !u64_ge(threshold,n) {m+=1u;}
    return f32(m)*bitcast<f32>(u32(127+(exp-odd)/2-23)<<23u);
}
fn fx_smooth(t:f32)->f32 {
    let x=clamp(t,0.0,1.0);
    return barrier((x*x)*barrier(3.0-barrier(2.0*x)));
}
// Ordered per-tile rectangle lists. Geometry and coverage bounds are prepared on the CPU.
fn fx_pattern(kind: u32, u: f32, v: f32, suppl: f32) -> f32 {
    let r2 = barrier(u*u) + barrier(v*v);
    let base = max(1.0-r2, 0.0);
    let sq = base*base;
    let bell = sq*sq;
    switch kind {
        case 0u: { return 1.0; }
        case 1u: { return select(0.0,1.0,r2<=bitcast<f32>(0x3f800001u)); }
        case 5u: { let a=max(1.0-fx_sqrt(r2),0.0); let b=a*a; return b*b; }
        case 14u: { return max(1.0-fx_sqrt(r2),0.0); }
        case 7u: {
            if suppl<=0.0 { return 1.0; }
            return fx_smooth(div_rn(1.0-abs(u),suppl))*fx_smooth(div_rn(1.0-abs(v),suppl));
        }
        case 2u: { return base; }
        case 3u: { return bell; }
        case 6u: { return 1.0-max(abs(u),abs(v)); }
        case 8u: { return ((u+1.0)*0.5)*(1.0-max(suppl,0.0)); }
        case 9u: {
            let waves=3.0+barrier(12.0*max(suppl,0.0));
            let cycle=rs_rem1(barrier(barrier(0.5*waves)*u));
            let half=min(cycle,1.0-cycle);
            let x=barrier(4.0*min(half,0.5-half));
            let a:f32=1.223936;
            // Rust evaluates each product left to right. Barriers also prevent
            // the driver from factoring/reassociating the quartic's powers.
            let quadratic=barrier(barrier(a*x)*x);
            let quartic=barrier(barrier(barrier(barrier((a-1.0)*x)*x)*x)*x);
            let wave=barrier(1.0-quadratic)+quartic;
            let carrier=select(-wave,wave,half<0.25);
            return barrier(base*barrier(carrier+1.0))*0.5;
        }
        case 10u: {
            let d=1.0-max(suppl,0.0);
            var factor=select(0.0,1.0,u<=0.0);
            if d>0.0 {factor=1.0-fx_smooth(div_rn(u,d));}
            if d<0.0 {factor=fx_smooth(div_rn(u-d,-d));}
            return bell*barrier(factor);
        }
        case 12u: {
            let d=1.0-max(suppl,0.0);
            var factor=0.0;
            if d!=0.0 {factor=1.0-fx_smooth(div_rn(u+1.0,2.0*d));}
            return 6.75*r2*sq*barrier(factor);
        }
        case 11u: { let t=1.0-abs(u); return bell*(t*t); }
        case 13u: {
            let a=max(abs(u)-barrier(0.77*max(suppl,0.0)),0.0);
            return max(1.0-barrier(20.0*(barrier(a*a)+barrier(v*v))),0.0);
        }
        default: { return barrier((u+1.0)*0.5)+suppl; }
    }
}
fn fx_half(v: f32) -> f32 {
    let bits=bitcast<u32>(v); let sign=bits&0x80000000u;
    let e=i32((bits>>23u)&255u)-127;
    if e==128 { return v; }
    if e>15 { return bitcast<f32>(sign|0x477fe000u); }
    if e< -24 { return bitcast<f32>(sign); }
    let shift=u32(23-min(10,24+e));
    return bitcast<f32>(bits & ~((1u<<shift)-1u));
}
fn fx_q8(v:f32)->u32 {
    return (u32(floor(clamp(v,0.0,1.0)*4096.0))*255u+2047u)>>12u;
}
// image.rs::half_sample: ties away, distinct from final half storage/truncation.
fn fx_half_sample(v:f32)->f32 {
    if !finite_bits(v) {return v;}
    let bits=bitcast<u32>(v); let sign=bits&0x80000000u;
    let e=i32((bits>>23u)&255u)-127;
    if e>15 {return bitcast<f32>(sign|0x477fe000u);}
    if e< -25 {return bitcast<f32>(sign);}
    if e== -25 {return bitcast<f32>(sign|(103u<<23u));}
    let drop=u32(23-min(10,24+e));
    return bitcast<f32>((bits+(1u<<(drop-1u)))&~((1u<<drop)-1u));
}
fn fx_fetch(d:vec4<u32>,x:i32,y:i32)->vec4<f32> {
    if x<0 || y<0 || x>=i32(d.y) || y>=i32(d.z) {return vec4<f32>(0.0);}
    return input_b[d.x+u32(y)*d.y+u32(x)];
}
fn fx_sample(d:vec4<u32>,lx:f32,ly:f32)->vec4<f32> {
    let u=(lx+1.0)*0.5; let v=(ly+1.0)*0.5;
    if (d.w>>8u)==2u {return fx_fetch(d,i32(floor(u*f32(d.y))),i32(floor(v*f32(d.z))));}
    let x=barrier(u*f32(d.y))-0.5; let y=barrier(v*f32(d.z))-0.5;
    let x0=floor(x); let y0=floor(y);
    let fx=rs_round((x-x0)*256.0)/256.0; let fy=rs_round((y-y0)*256.0)/256.0;
    let cross=rs_round(fx*fy*256.0)/256.0;
    let wt=vec4<f32>(barrier(1.0-fx-fy+cross),barrier(fx-cross),barrier(fy-cross),cross);
    let a=fx_fetch(d,i32(x0),i32(y0)); let b=fx_fetch(d,i32(x0)+1,i32(y0));
    let c=fx_fetch(d,i32(x0),i32(y0)+1); let e=fx_fetch(d,i32(x0)+1,i32(y0)+1);
    if (d.w&3u)<=1u {
        let q=vec4<u32>(wt*256.0);
        let code=(b_code16(a)*q.x+b_code16(b)*q.y+b_code16(c)*q.z+b_code16(e)*q.w+vec4<u32>(128u))>>vec4<u32>(8u);
        return vec4<f32>(unorm_float(code.x,16u),unorm_float(code.y,16u),unorm_float(code.z,16u),unorm_float(code.w,16u));
    }
    var out:vec4<f32>;
    for(var ch=0u;ch<4u;ch+=1u) {
        out[ch]=b_weighted(a[ch],b[ch],c[ch],e[ch],wt);
        if (d.w&3u)==2u {out[ch]=fx_half_sample(out[ch]);}
    }
    return out;
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
    let w=config.header.z; let h=config.header.y/w;
    if id.x>=w || id.y>=h { return; }
    let k=id.y*w+id.x;
    let tile=(id.y/16u)*config.selectors.x+id.x/16u;
    let list=bitcast<vec4<u32>>(input_b[tile]);
    var dst=input_a[k];
    for(var i=0u; i<list.y; i+=1u) {
        let r=bitcast<vec4<u32>>(input_b[list.x+i]).x;
        let bounds=bitcast<vec4<u32>>(input_b[r]);
        if id.x<bounds.x || id.x>bounds.y || id.y<bounds.z || id.y>bounds.w { continue; }
        let cover=bitcast<vec4<u32>>(input_b[r+5u]);
        if cover.y!=0u {
            let row=bitcast<vec4<u32>>(input_b[cover.x+id.y-bounds.z]);
            if id.x<row.x || id.x>=row.y { continue; }
        }
        let geo=input_b[r+1u];
        let flags=bitcast<vec4<u32>>(input_b[r+2u]);
        let op=input_b[r+3u];
        let params=input_b[r+4u];
        var dx=(f32(i32(id.x)+bitcast<i32>(flags.x))+0.5)/f32(w)-geo.x;
        var dy=(f32(i32(id.y)+bitcast<i32>(flags.y))+0.5)/f32(h)-geo.y;
        if config.header.x!=0u {
            let roi=config.parameters[0];
            dx=(roi.x+barrier((f32(id.x)+0.5)/f32(w)*roi.z))-geo.x;
            dy=(roi.y+barrier((f32(id.y)+0.5)/f32(h)*roi.w))-geo.y;
        }
        let lx=div_rn(barrier(params.w*dx)-barrier(params.z*dy),geo.z);
        let ly=div_rn(barrier(params.z*dx)+barrier(params.w*dy),geo.w);
        var sample=vec4<f32>(1.0,1.0,1.0,0.0);
        let desc=bitcast<vec4<u32>>(input_b[r+6u]);
        if flags.z==16u {sample=fx_sample(desc,lx,ly);}
        else {sample.w=fx_pattern(flags.z,lx,ly,params.y);}
        var p=sample.w;
        let weight=params.x; let suppl=params.y; let bm=flags.w!=0u;
        if config.options.y!=0u {
            let a=op.w*p*weight;
            if config.selectors.y==0u {
                let s=vec4<f32>(unorm_float(fx_q8(op.x*sample.x*a),8u),unorm_float(fx_q8(op.y*sample.y*a),8u),unorm_float(fx_q8(op.z*sample.z*a),8u),unorm_float(fx_q8(a),8u));
                for(var ch=0u; ch<4u; ch+=1u) {
                    var d=dst[ch];
                    if bm || ch==3u { d=barrier(d*(1.0-s.w)); }
                    dst[ch]=unorm_float(u32(floor(barrier(clamp(s[ch]+d,0.0,1.0)*255.0)+0.5)),8u);
                }
            } else if flags.z==15u && !bm {
                let core=op.w*(p-suppl)*weight;
                for(var ch=0u; ch<3u; ch+=1u) { dst[ch]+=barrier(op[ch]*core)+suppl; }
                dst.w+=barrier((core+suppl)*(1.0-dst.w));
            } else {
                for(var ch=0u; ch<3u; ch+=1u) {
                    let c=barrier(op[ch]*sample[ch]);
                    if bm { dst[ch]+=barrier((c-dst[ch])*a); }
                    else { dst[ch]+=barrier(c*a); }
                }
                dst.w+=barrier(a*(1.0-dst.w));
            }
        } else {
            if flags.z==16u {
                p=0.0;
                if (desc.w&16u)==0u {p=1.0-barrier(1.0-sample.x);}
            }
            var val=op.x*p*weight;
            if flags.z==15u { val=barrier(op.x*(p-suppl)*weight)+suppl; }
            var g=dst.x+val;
            if bm { g=max(dst.x,val); }
            if config.selectors.y==0u {
                let s=fx_q8(val); let d=u32(floor(barrier(dst.x*255.0)+0.5));
                var q=min(d+s,255u); if bm { q=max(d,s); }
                g=unorm_float(q,8u);
            }
            dst=vec4<f32>(g,g,g,1.0);
        }
    }
    if config.options.x==2u { dst=vec4<f32>(fx_half(dst.x),fx_half(dst.y),fx_half(dst.z),fx_half(dst.w)); }
    store(k,dst);
}
