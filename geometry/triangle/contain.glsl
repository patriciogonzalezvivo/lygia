#include "triangle.glsl"

/*
contributors: Thomas Müller & Alex Evans
description: Does the position lie within the triangle
use: <vec3> contain(<Triangle> tri, <vec3> _pos) 
*/

#ifndef FNC_TRIANGLE_CONTAIN
#define FNC_TRIANGLE_CONTAIN

bool contain(Triangle _tri, vec3 _pos) { 
    // Move the triangle so that the point becomes the
    // triangles origin
    vec3 local_a = _tri.a - _pos;
    vec3 local_b = _tri.b - _pos;
    vec3 local_c = _tri.c - _pos;

    // The point should be moved too, so they are both
    // relative, but because we don't use p in the
    // equation anymore, we don't need it!
    // p -= p;

    // Compute the normal vectors for triangles:
    // u = normal of PBC
    // v = normal of PCA
    // w = normal of PAB

    vec3 u = cross(local_b, local_c);
    vec3 v = cross(local_c, local_a);
    vec3 w = cross(local_a, local_b);

    // Test to see if the normals are facing the same direction.
    // If yes, the point is inside, otherwise it isn't.
    return dot(u, v) >= 0.0 && dot(u, w) >= 0.0;
}

#endif