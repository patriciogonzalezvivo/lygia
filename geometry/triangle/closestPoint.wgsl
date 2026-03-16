#include "triangle.wgsl"
#include "barycentric.wgsl"

/*
contributors: 
description: Returns the closest point on the surface of a triangle
use: <vec3> closestDistance(<Triangle> tri, <vec3> _pos) 
*/

// https://github.com/nmoehrle/libacc/blob/master/primitives.h#L71
fn closestPoint(_tri: Triangle, _triNormal: vec3f, _pos: vec3f) -> vec3f {
    let ab = _tri.b - _tri.a;
    let ac = _tri.c - _tri.a;
    let normal = _triNormal;

    let p = _pos - dot(_triNormal, _pos - _tri.a) * _triNormal;
    let ap = p - _tri.a;

    let bcoords = barycentric(ab, ac, ap);

    if (bcoords.x < 0.0f) {
        let bc = _tri.c - _tri.b;
        let n = length( bc );
        let t = max(0.0f, min( dot(bc, p - _tri.b)/n, n));
        return _tri.b + t / n * bc;
    }

    if (bcoords.y < 0.0f) {
        let ca = _tri.a - _tri.c;
        let n = length( ca );
        let t = max(0.0f, min( dot(ca, p - _tri.c)/n, n));
        return _tri.c + t / n * ca;
    }

    if (bcoords.z < 0.0f) {
        let n = length( ab );
        let t = max(0.0f, min( dot(ab, p - _tri.a)/n, n));
        return _tri.a + t / n * ab;
    }

    return (_tri.a * bcoords.x + _tri.b * bcoords.y + _tri.c * bcoords.z);
}

// https://github.com/nmoehrle/libacc/blob/master/primitives.h#L71
fn closestPointa(_tri: Triangle, _pos: vec3f) -> vec3f {
    let ab = _tri.b - _tri.a;
    let ac = _tri.c - _tri.a;
    let normal = normalize( cross(ac,ab) );

    let p = _pos - dot(normal, _pos - _tri.a) * normal;
    let ap = p - _tri.a;

    let bcoords = barycentric(ab, ac, ap);

    if (bcoords.x < 0.0f) {
        let bc = _tri.c - _tri.b;
        let n = length( bc );
        let t = max(0.0f, min( dot(bc, p - _tri.b)/n, n));
        return _tri.b + t / n * bc;
    }

    if (bcoords.y < 0.0f) {
        let ca = _tri.a - _tri.c;
        let n = length( ca );
        let t = max(0.0f, min( dot(ca, p - _tri.c)/n, n));
        return _tri.c + t / n * ca;
    }

    if (bcoords.z < 0.0f) {
        let n = length( ab );
        let t = max(0.0f, min( dot(ab, p - _tri.a)/n, n));
        return _tri.a + t / n * ab;
    }

    return (_tri.a * bcoords.x + _tri.b * bcoords.y + _tri.c * bcoords.z);
}
