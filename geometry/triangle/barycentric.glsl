#include "triangle.glsl"
#include "area.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns the centroid of a triangle
use: <vec3> centroid(<Triangle> tri)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_TRIANGLE_BARYCENTRIC
#define FNC_TRIANGLE_BARYCENTRIC

vec3 barycentric(vec3 _a, vec3 _b, vec3 _c ) {
    /* Derived from the book "Real-Time Collision Detection"
     * by Christer Ericson published by Morgan Kaufmann in 2005 */
    float daa = dot(_a, _a);
    float dab = dot(_a, _b);
    float dbb = dot(_b, _b);
    float dca = dot(_c, _a);
    float dcb = dot(_c, _b);
    float denom = daa * dbb - dab * dab;
    float y = (dbb * dca - dab * dcb) / denom;
    float z = (daa * dcb - dab * dca) / denom;
    return vec3( 1.0f - y - z, y, z);
}

vec3 barycentric(Triangle _tri) { return barycentric(_tri.a, _tri.b, _tri.c); }

vec3 barycentric(Triangle _tri, vec3 _pos) {
    vec3 f0 = _tri.a - _pos;
    vec3 f1 = _tri.b - _pos;
    vec3 f2 = _tri.c - _pos;

    return vec3( length(cross(f1, f2)),                      // p1's triangle area / a
                        length(cross(f2, f0)),                      // p2's triangle area / a 
                        length(cross(f0, f1)) ) / area(_tri) ;      // p3's triangle area / a
}

#endif
