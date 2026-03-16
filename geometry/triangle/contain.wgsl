#include "triangle.wgsl"

/*
contributors: Thomas Müller & Alex Evans
description: Does the position lie within the triangle
use: <vec3> contain(<Triangle> tri, <vec3> _pos) 
*/

fn contain(_tri: Triangle, _pos: vec3f) -> bool {
    // Move the triangle so that the point becomes the
    // triangles origin
    let local_a = _tri.a - _pos;
    let local_b = _tri.b - _pos;
    let local_c = _tri.c - _pos;

    // The point should be moved too, so they are both
    // relative, but because we don't use p in the
    // equation anymore, we don't need it!
    // p -= p;

    // Compute the normal vectors for triangles:
    // u = normal of PBC
    // v = normal of PCA
    // w = normal of PAB

    let u = cross(local_b, local_c);
    let v = cross(local_c, local_a);
    let w = cross(local_a, local_b);

    // Test to see if the normals are facing the same direction.
    // If yes, the point is inside, otherwise it isn't.
    return dot(u, v) >= 0.0 && dot(u, w) >= 0.0;
}
