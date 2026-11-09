// image_pixel (packed or two-slot input access) is in gpu_prelude.wgsl.
//
// Every kernel here mirrors one CPU function of ops.rs operation by operation, so that the
// results agree bit for bit (ops.rs "blend", "levels", "normal", "sharpen"). The driver
// contracts a product into the addition that follows it (a*b + c) and distributes a constant
// factor over a sum (c*(a + b) becomes c*a + c*b), so every product that feeds an addition and
// every sum that feeds a product is wrapped in barrier() (gpu_prelude.wgsl); the divisions and
// the square root go through the correctly rounded div_rn and sqrt_rn.
//
// The pow of the levels kernel is the one exception: the CPU uses Rust's libm powf, which is
// not correctly rounded, so the GPU's pow differs from it by about one ULP (documented in
// cx/gpu_exact/REPORT.md).

// The blend function of ops.rs "blend": f(b, f) per mode.
fn blend_channel(b:f32,f:f32,mode:u32)->f32 {
    switch mode {
        case 1u: {return b+f;}
        case 2u: {return b-f;}
        case 3u: {return b*f;}
        case 4u: {return b+barrier(2.0*f)-1.0;}
        case 5u: {return rs_max(b,f);}
        case 6u: {return rs_min(b,f);}
        case 8u: {
            // 0/0 is 1 on the reference GPU (ops.rs "blend": a NaN quotient becomes 1)
            let q=div_rn(b,f);
            if nan_bits(q) {return 1.0;}
            return q;
        }
        case 9u: {if b<0.5 {return barrier(barrier(2.0*b)*f);} return 1.0-barrier(barrier(2.0*(1.0-b))*(1.0-f));}
        case 10u: {return 1.0-barrier((1.0-b)*(1.0-f));}
        case 11u: {
            if f<0.5 {return barrier(barrier(2.0*b)*f)+barrier(barrier(b*b)*(1.0-barrier(2.0*f)));}
            return barrier(barrier(2.0*b)*(1.0-f))+barrier(sqrt_rn(b)*barrier(2.0*f-1.0));
        }
        default: {return f;}
    }
}
// The GPU's mix: b·(1 − t) + f·t (ops.rs "blend", lerp). The sum 1 − t feeds a product: the
// driver would distribute the factor b over it.
fn mix_layers(a:f32,b:f32,t:f32)->f32 {return barrier(a*barrier(1.0-t))+barrier(b*t);}
fn blend_pixel(k:u32)->vec4<f32> {
    let b=image_pixel(0u,k);let f=image_pixel(1u,k);
    let x=k%config.header.z;let y=k/config.header.z;
    let crop=config.selectors;
    if config.values.y!=0.0 && (x<crop.x || y<crop.y || x>=crop.z || y>=crop.w) {return b;}
    var opacity=config.values.x;
    if config.options.w>2u {opacity=barrier(opacity*image_pixel(2u,k).x);}
    opacity=rs_clamp(opacity,0.0,1.0);
    let mode=config.header.w&15u;let alpha_mode=(config.header.w>>11u)&3u;
    if config.options.y==0u {return vec4<f32>(mix_layers(b.x,blend_channel(b.x,f.x,mode),opacity));}
    var out:vec4<f32>;
    if mode==7u {
        // switch: every channel from background to foreground by the opacity
        for(var c=0u;c<4u;c++) {out[c]=mix_layers(b[c],f[c],opacity);}
        return out;
    }
    let ba=b.w;let fa=f.w;
    if alpha_mode==1u {
        // ignore alpha
        for(var c=0u;c<3u;c++) {out[c]=mix_layers(b[c],blend_channel(b[c],f[c],mode),opacity);}
        out.w=ba+barrier(barrier(opacity*fa)*(1.0-ba));
    } else if alpha_mode==2u {
        // straight: the W3C compositing of the two layers, mixed by the opacity
        let full=fa+barrier(ba*(1.0-fa));
        let a=barrier(ba*(1.0-opacity))+barrier(full*opacity);
        for(var c=0u;c<3u;c++) {
            let mixed=barrier(barrier(barrier(blend_channel(b[c],f[c],mode)*fa)*ba)
                              +barrier(barrier(f[c]*fa)*(1.0-ba)))
                     +barrier(barrier(b[c]*ba)*(1.0-fa));
            let pre=barrier(barrier(b[c]*ba)*(1.0-opacity))+barrier(mixed*opacity);
            out[c]=0.0;if a!=0.0 {out[c]=div_rn(pre,a);}
        }
        out.w=a;
    } else if alpha_mode==3u && mode==0u {
        // premultiplied copy
        let faop=barrier(fa*opacity);
        for(var c=0u;c<3u;c++) {out[c]=barrier(f[c]*opacity)+barrier(b[c]*(1.0-faop));}
        out.w=faop+barrier(ba*(1.0-faop));
    } else {
        // source alpha (the premultiplied modes other than copy behave the same)
        let faop=barrier(fa*opacity);
        for(var c=0u;c<3u;c++) {out[c]=mix_layers(b[c],blend_channel(b[c],f[c],mode),faop);}
        out.w=ba+barrier(barrier(opacity*fa)*(1.0-ba));
    }
    return out;
}
fn levels_pixel(k:u32)->vec4<f32> {
    let p=input_a[k];var out:vec4<f32>;
    let il=config.parameters[0];let ih=config.parameters[1];let exponent=config.parameters[2];
    let ol=config.parameters[3];let oh=config.parameters[4];
    for(var c=0u;c<4u;c++) {
        // inlow == inhigh: the range becomes 1e-4
        var d=0.0001;if ih[c]!=il[c] {d=ih[c]-il[c];}
        var x=div_rn(p[c]-il[c],d);
        if config.header.w==0u {x=rs_clamp(x,0.0,1.0);}
        // the exponent is 1 for the default midpoint: no pow then
        if exponent[c]!=1.0 {x=pow(rs_max(x,0.0),exponent[c]);}
        var y=ol[c]+barrier(x*barrier(oh[c]-ol[c]));
        if config.header.w==0u {y=rs_clamp(y,0.0,1.0);}
        out[c]=y;
    }
    return out;
}
fn neighbor(k:u32,dx:i32,dy:i32)->vec4<f32> {
    let w=config.header.z;let h=config.header.y/w;
    let x=i32(k%w)+dx;let y=i32(k/w)+dy;
    var xx=u32(clamp(x,0i,i32(w)-1i));var yy=u32(clamp(y,0i,i32(h)-1i));
    // the tiled axes wrap (the node's width and height are powers of two)
    if (config.header.w&1u)!=0u {xx=bitcast<u32>(x)&(w-1u);}
    if (config.header.w&2u)!=0u {yy=bitcast<u32>(y)&(h-1u);}
    return input_a[yy*w+xx];
}
// v · 0.5 + 0.5, the normal map's encoding of a unit vector.
fn half_plus(v:f32)->f32 {return barrier(v*0.5)+0.5;}
fn normal_pixel(k:u32)->vec4<f32> {
    let c=input_a[k].x;
    let dx=c-neighbor(k,-1,0).x;let dy=c-neighbor(k,0,-1).x;
    let vx=-dx*config.values.x;let vy=-dy*config.values.y;
    let length=sqrt_rn(barrier(barrier(vx*vx)+barrier(vy*vy))+1.0);
    let alpha=select(1.0,c,(config.header.w&4u)!=0u);
    return vec4<f32>(half_plus(div_rn(vx,length)),half_plus(div_rn(vy,length)),half_plus(div_rn(1.0,length)),alpha);
}
// One channel of ops.rs "sharpen": c + intensity · ((2c − l − r) · pixelsize.x + (2c − u − d) ·
// pixelsize.y).
fn sharpen_channel(c:f32,l:f32,r:f32,u:f32,d:f32)->f32 {
    let hx=barrier(barrier(barrier(2.0*c)-l)-r);
    let hy=barrier(barrier(barrier(2.0*c)-u)-d);
    let s=barrier(barrier(hx*config.values.y)+barrier(hy*config.values.z));
    return c+barrier(config.values.x*s);
}
fn sharpen_pixel(k:u32)->vec4<f32> {
    let c=input_a[k];
    let l=neighbor(k,-1,0);let r=neighbor(k,1,0);let u=neighbor(k,0,-1);let d=neighbor(k,0,1);
    return vec4<f32>(
        sharpen_channel(c.x,l.x,r.x,u.x,d.x),
        sharpen_channel(c.y,l.y,r.y,u.y,d.y),
        sharpen_channel(c.z,l.z,r.z,u.z,d.z),
        sharpen_channel(c.w,l.w,r.w,u.w,d.w));
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.y * config.header.z + id.x;
    if id.x >= config.header.z || k >= config.header.y { return; }
    if config.header.x==7u {store(k,s_resize_pixel(k));return;}
    if config.header.x==5u {store(k,normal_pixel(k));return;}
    if config.header.x==6u {store(k,sharpen_pixel(k));return;}
    if config.header.x==3u {store(k,levels_pixel(k));return;}
    if config.header.x==4u {store(k,blend_pixel(k));return;}
    if config.header.x == 0u {
        store(k, config.values);
    } else if config.header.x == 1u {
        let a = input_a[k];
        let b = input_b[k];
        var v: vec4<f32>;
        for (var c = 0u; c < 4u; c++) {
            let s = config.selectors[c];
            if s < 4u { v[c] = a[s]; } else { v[c] = b[(s - 4u) & 3u]; }
        }
        store(k, v);
    } else {
        let p = input_a[k];
        let w = config.values;
        // ((p0*w0 + p1*w1) + p2*w2) + p3*w3 and lerp(matte, g, alpha), rounded as on the CPU.
        var g = ((barrier(p.x * w.x) + barrier(p.y * w.y)) + barrier(p.z * w.z)) + barrier(p.w * w.w);
        if config.header.w != 0u { g = config.extra.x + barrier((g - config.extra.x) * p.w); }
        store(k, vec4<f32>(g, g, g, 1.0));
    }
}
