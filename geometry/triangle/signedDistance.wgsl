#include "triangle.wgsl"
#include "normal.wgsl"
#include "closestPoint.wgsl"

/*
contributors: 
description: Returns the signed distance from the surface of a triangle to a point
use: <vec3> closestDistance(<Triangle> tri, <vec3> _pos) 
*/

fn signedDistance(_tri: Triangle, _triNormal: vec3f, _p: vec3f) -> f32 {
    let nearest = closestPoint(_tri, _triNormal, _p);
    let delta = _p - nearest;
    let distance = length(delta);
    distance *= sign( dot(delta/distance, _triNormal) );
    return distance;
}

fn signedDistancea(_tri: Triangle, _p: vec3f) -> f32 { return signedDistance(_tri, normal(_tri), _p); }
