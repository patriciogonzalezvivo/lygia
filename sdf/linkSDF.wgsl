/*
contributors:  Inigo Quiles
description: generate the SDF of a link
use: <float> linkSDF( <vec3> p, <float> le, <float> r1, <float> r2 ) 
*/

fn linkSDF(p: vec3f, le: f32, r1: f32, r2: f32) -> f32 {
    let q = vec3f( p.x, max(abs(p.y)-le,0.0), p.z );
    return length(vec2f(length(q.xy)-r1,q.z)) - r2;
}
