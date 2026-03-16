/*
contributors:  Inigo Quiles
description: generate the SDF of a tetrahedron
use: <float> tetrahedronSDF( in <vec3> pos, in <float> h ) 
*/

fn tetrahedronSDF(p: vec3f, h: f32) -> f32 {
    let q = abs(p);
    
    let y = p.y;
    let d1 = q.z-max(0.,y);
    let d2 = max(q.x*.5 + y*.5,.0) - min(h, h+y);
    return length(max(vec2f(d1,d2),.005)) + min(max(d1,d2), 0.0);
}
