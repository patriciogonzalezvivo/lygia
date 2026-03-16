/*
contributors:  Inigo Quiles
description: repeat operation for 2D/3D SDFs 
use: <vec4> opElongate( in <vec3> p, in <vec3> h )
*/

fn opRepeat2(p: vec2f, s: f32) -> vec2f {
    return mod(p+s*0.5,s)-s*0.5;
}

fn opRepeat3(p: vec3f, c: vec3f) -> vec3f {
    return mod(p+0.5*c,c)-0.5*c;
}

fn opRepeat2a(p: vec2f, lima: vec2f, limb: vec2f, s: f32) -> vec2f {
    return p-s*clamp(floor(p/s),lima,limb);
}

fn opRepeat3a(p: vec3f, lima: vec3f, limb: vec3f, s: f32) -> vec3f {
    return p-s*clamp(floor(p/s),lima,limb);
}
