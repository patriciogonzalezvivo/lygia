#include "triangle.wgsl"
#include "area.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns the barycentric coordinates of a point or triangle
use: <vec3> barycentric(<Triangle> tri [, <vec3> pos])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn barycentric3(_a: vec3f, _b: vec3f, _c: vec3f) -> vec3f {
    /* Derived from the book "Real-Time Collision Detection"
     * by Christer Ericson published by Morgan Kaufmann in 2005 */
    let daa = dot(_a, _a);
    let dab = dot(_a, _b);
    let dbb = dot(_b, _b);
    let dca = dot(_c, _a);
    let dcb = dot(_c, _b);
    let denom = daa * dbb - dab * dab;
    let y = (dbb * dca - dab * dcb) / denom;
    let z = (daa * dcb - dab * dca) / denom;
    return vec3f( 1.0f - y - z, y, z);
}

fn barycentric(_tri: Triangle) -> vec3f { return barycentric(_tri.a, _tri.b, _tri.c); }

fn barycentrica(_tri: Triangle, _pos: vec3f) -> vec3f {
    let f0 = _tri.a - _pos;
    let f1 = _tri.b - _pos;
    let f2 = _tri.c - _pos;

    return vec3f( length(cross(f1, f2)),                      // p1's triangle area / a
                        length(cross(f2, f0)),                      // p2's triangle area / a 
                        length(cross(f0, f1)) ) / area(_tri) ;      // p3's triangle area / a
}
