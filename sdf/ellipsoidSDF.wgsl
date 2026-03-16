/*
contributors:  Inigo Quiles
description: generate the SDF of an approximated ellipsoid
use: <float> ellipsoidSDF( in <vec3> p, in <vec3> r )
*/

fn ellipsoidSDF(p: vec3f, r: vec3f) -> f32 {
    let k0 = length(p/r);
    let k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}
