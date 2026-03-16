#include "../math/const.wgsl"

/*
description: generate the SDF of a dodecahedron
use: <float> dodecahedronSDF( in <vec3> pos [, in <float> size] ) 
*/

fn dodecahedronSDF3(p: vec3f) -> f32 {
    let n = normalize(vec3f(PHI,1.0,0.0));
    p = abs(p);
    let a = dot(p,n.xyz);
    let b = dot(p,n.zxy);
    let c = dot(p,n.yzx);
    // return max(max(a,b),c)-PHI*n.y;
    return (max(max(a,b),c)-n.x);
}

fn dodecahedronSDF3a(p: vec3f, radius: f32) -> f32 {
    let n = normalize(vec3f(PHI,1.0,0.0));

    p = abs(p / radius);
    let a = dot(p, n.xyz);
    let b = dot(p, n.zxy);
    let c = dot(p, n.yzx);
    return (max(max(a,b),c)-n.x) * radius;
}
