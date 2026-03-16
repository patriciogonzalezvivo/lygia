/*
contributors:  Inigo Quiles
description: generate the SDF of a hexagonal prism
use: <float> hexPrismSDF( in <vec3> pos, in <vec2> h ) 
*/

fn hexPrismSDF(p: vec3f, h: vec2f) -> f32 {
    let q = abs(p);
    let d1 = q.z-h.y;
    let d2 = max((q.x*0.866025+q.y*0.5),q.y)-h.x;
    return length(max(vec2f(d1,d2),0.0)) + min(max(d1,d2), 0.);
}
