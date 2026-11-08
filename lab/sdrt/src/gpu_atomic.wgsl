// image_pixel (packed or two-slot input access) is in gpu_prelude.wgsl.
fn blend_channel(b:f32,f:f32,mode:u32)->f32 {
    switch mode {
        case 1u: {return b+f;}
        case 2u: {return b-f;}
        case 3u: {return b*f;}
        case 4u: {return b+2.0*f-1.0;}
        case 5u: {return max(b,f);}
        case 6u: {return min(b,f);}
        case 8u: {
            if b==0.0 && f==0.0 {return 1.0;}
            let q=b/f;
            if (bitcast<u32>(q)&0x7fffffffu)>0x7f800000u {return 1.0;}
            return q;
        }
        case 9u: {if b<0.5 {return 2.0*b*f;} return 1.0-2.0*(1.0-b)*(1.0-f);}
        case 10u: {return 1.0-(1.0-b)*(1.0-f);}
        case 11u: {if f<0.5 {return 2.0*b*f+b*b*(1.0-2.0*f);} return 2.0*b*(1.0-f)+sqrt(b)*(2.0*f-1.0);}
        default: {return f;}
    }
}
fn mix_layers(a:f32,b:f32,t:f32)->f32 {return a*(1.0-t)+b*t;}
fn blend_pixel(k:u32)->vec4<f32> {
    let b=image_pixel(0u,k);let f=image_pixel(1u,k);
    let x=k%config.header.z;let y=k/config.header.z;
    let crop=config.selectors;
    if config.values.y!=0.0 && (x<crop.x || y<crop.y || x>=crop.z || y>=crop.w) {return b;}
    var opacity=config.values.x;
    if config.options.w>2u {opacity*=image_pixel(2u,k).x;}
    opacity=clamp(opacity,0.0,1.0);
    let mode=config.header.w&15u;let alpha_mode=(config.header.w>>11u)&3u;
    if config.options.y==0u {return vec4<f32>(mix_layers(b.x,blend_channel(b.x,f.x,mode),opacity));}
    var out:vec4<f32>;
    if mode==7u {return b*(1.0-opacity)+f*opacity;}
    let ba=b.w;let fa=f.w;
    if alpha_mode==1u {
        for(var c=0u;c<3u;c++) {out[c]=mix_layers(b[c],blend_channel(b[c],f[c],mode),opacity);}
        out.w=ba+opacity*fa*(1.0-ba);
    } else if alpha_mode==2u {
        let full=fa+ba*(1.0-fa);let a=ba*(1.0-opacity)+full*opacity;
        for(var c=0u;c<3u;c++) {
            let mixed=(blend_channel(b[c],f[c],mode)*fa*ba+f[c]*fa*(1.0-ba))+b[c]*ba*(1.0-fa);
            let pre=b[c]*ba*(1.0-opacity)+mixed*opacity;
            out[c]=0.0;if a!=0.0 {out[c]=pre/a;}
        }
        out.w=a;
    } else if alpha_mode==3u && mode==0u {
        out=f*opacity+b*(1.0-fa*opacity);
    } else {
        for(var c=0u;c<3u;c++) {out[c]=mix_layers(b[c],blend_channel(b[c],f[c],mode),fa*opacity);}
        out.w=ba+opacity*fa*(1.0-ba);
    }
    return out;
}
fn levels_pixel(k:u32)->vec4<f32> {
    let p=input_a[k];var out:vec4<f32>;
    let il=config.parameters[0];let ih=config.parameters[1];let exponent=config.parameters[2];
    let ol=config.parameters[3];let oh=config.parameters[4];
    for(var c=0u;c<4u;c++) {
        var d=0.0001;if ih[c]!=il[c] {d=ih[c]-il[c];}
        var x=(p[c]-il[c])/d;
        if config.header.w==0u {x=clamp(x,0.0,1.0);}
        if exponent[c]!=1.0 {x=pow(max(x,0.0),exponent[c]);}
        var y=ol[c]+x*(oh[c]-ol[c]);
        if config.header.w==0u {y=clamp(y,0.0,1.0);}
        out[c]=y;
    }
    return out;
}
fn neighbor(k:u32,dx:i32,dy:i32)->vec4<f32> {
    let w=config.header.z;let h=config.header.y/w;
    let x=i32(k%w)+dx;let y=i32(k/w)+dy;
    var xx=u32(clamp(x,0i,i32(w)-1i));var yy=u32(clamp(y,0i,i32(h)-1i));
    if (config.header.w&1u)!=0u {xx=bitcast<u32>(x)&(w-1u);}
    if (config.header.w&2u)!=0u {yy=bitcast<u32>(y)&(h-1u);}
    return input_a[yy*w+xx];
}
fn normal_pixel(k:u32)->vec4<f32> {
    let c=input_a[k].x;
    let dx=c-neighbor(k,-1,0).x;let dy=c-neighbor(k,0,-1).x;
    let vx=-dx*config.values.x;let vy=-dy*config.values.y;
    let length=sqrt((vx*vx+vy*vy)+1.0);
    let alpha=select(1.0,c,(config.header.w&4u)!=0u);
    return vec4<f32>(vx/length*0.5+0.5,vy/length*0.5+0.5,1.0/length*0.5+0.5,alpha);
}
fn sharpen_pixel(k:u32)->vec4<f32> {
    let c=input_a[k];
    let l=neighbor(k,-1,0);let r=neighbor(k,1,0);let u=neighbor(k,0,-1);let d=neighbor(k,0,1);
    return c+config.values.x*(((2.0*c-l)-r)*config.values.y+((2.0*c-u)-d)*config.values.z);
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.y * config.header.z + id.x;
    if id.x >= config.header.z || k >= config.header.y { return; }
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
