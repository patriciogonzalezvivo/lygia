/*
contributors:  Inigo Quiles
description: revolve operation of a 2D SDFs into a 3D one
use: <vec2> opRevolve( in <vec3> p, <float> w ) 
*/

fn opRevolve(p: vec3f, w: f32) -> vec2f {
    return vec2f( length(p.xz) - w, p.y );
}
