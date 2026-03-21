#include "triangle.wgsl"
#include "../../lighting/ray.wgsl"

/*
contributors: Inigo Quiles
description: Based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
use: <float> intersect(<Triangle> tri, <vec3> rayOrigin, <vec3> rayDir, out <vec3> intersectionPoint)
*/

fn intersect(_tri: Triangle, _rayOrigin: vec3f, _rayDir: vec3f, _point: vec3f) -> f32 {
    let v1v0 = _tri.b - _tri.a;
    let v2v0 = _tri.c - _tri.a;
    let rov0 = _rayOrigin - _tri.a;
    _point = cross(v1v0, v2v0);
    let q = cross(rov0, _rayDir);
    let d = 1.0f / dot(_rayDir, _point);
    let u = d * -dot(q, v2v0);
    let v = d *  dot(q, v1v0);
    let t = d * -dot(_point, rov0);
    if (u < 0.0f || u > 1.0f || v < 0.0f || (u+v) > 1.0f || t < 0.0f)
        t = 9999999.9f; // No intersection

    return t;
}

fn intersecta(_tri: Triangle, _ray: Ray, _point: vec3f) -> f32 { return intersect(_tri, _ray.origin, _ray.direction, _point); }
fn intersectb(_tri: Triangle, _ray: Ray) -> f32 {
    var p: vec3f;
    return intersect(_tri, _ray, p); 
}
