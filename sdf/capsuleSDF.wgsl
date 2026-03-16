#include "../math/saturate.wgsl"
/*
contributors:  Inigo Quiles
description: generate a SDF of a capsule
use: <float> capusleSDF( in <vec3> pos, in <vec3> a, <vec3> b, <float> r ) 
*/

fn capsuleSDF(p: vec3f, a: vec3f, b: vec3f, r: f32) -> f32 {
    let pa = p-a, ba = b-a;
    let h = saturate( dot(pa,ba)/dot(ba,ba) );
    return length( pa - ba*h ) - r;
}
