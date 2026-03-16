/*
contributors:  Inigo Quiles
description: elongate operation of two SDFs 
use: <vec4> opElongate( in <vec3> p, in <vec3> h )
*/

fn opElongate2(p: vec2f, h: vec2f) -> vec2f {
    return p-clamp(p,-h,h); 
}

fn opElongate3(p: vec3f, h: vec3f) -> vec3f {
    return p-clamp(p,-h,h); 
}

fn opElongate4(p: vec4f, h: vec4f) -> vec4f {
    let q = abs(p)-h;
    return vec4f( max(q,0.0), min(max(q.x,max(q.y,q.z)), 0.0) );
}
