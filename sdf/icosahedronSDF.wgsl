#include "../math/const.wgsl"

/*
description: generate the SDF of a icosahedron
use: <float> icosahedronSDF( in <vec3> pos, in <float> size ) 
*/

fn icosahedronSDF(p: vec3f, radius: f32) -> f32 {
    let q = 2.61803398875; // Golden Ratio + 1 = (sqrt(5)+3)/2;
    let n1 = normalize(vec3f(q, 1,0));
    let n2 = vec3f(0.57735026919);  // = sqrt(3)/3);

    p = abs(p / radius);
    let a = dot(p, n1.xyz);
    let b = dot(p, n1.zxy);
    let c = dot(p, n1.yzx);
    let d = dot(p, n2) - n1.x;
    return max(max(max(a,b),c)-n1.x,d) * radius;
}
