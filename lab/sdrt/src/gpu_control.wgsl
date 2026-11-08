// Scalar slots have a dynamic type because different branches can store different kinds.
// Tags follow gpu_pp::Kind: float=0, integer=1, bool=2.
fn var_f32(slot:u32)->f32 {
    let v=variables[slot];
    if v.y==1u {return f32(bitcast<i32>(v.x));}
    if v.y==2u {return select(0.0,1.0,v.x!=0u);}
    return bitcast<f32>(v.x);
}
fn var_i32(slot:u32)->i32 {
    let v=variables[slot];
    if v.y==1u {return bitcast<i32>(v.x);}
    if v.y==2u {return select(0i,1i,v.x!=0u);}
    return to_int(bitcast<f32>(v.x));
}
fn var_bool(slot:u32)->bool {
    let v=variables[slot];
    if v.y==0u {return bitcast<f32>(v.x)!=0.0;}
    return v.x!=0u;
}
