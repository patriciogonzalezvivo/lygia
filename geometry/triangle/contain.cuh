#include "triangle.cuh"
#include "../../math/operations.cuh"
#include "../../math/cross.cuh"

/*
contributors: Thomas Müller & Alex Evans
description: Does the position lie within the triangle
use: <float3> contain(<Triangle> tri, <float3> _pos) 
*/

#ifndef FNC_TRIANGLE_CONTAIN
#define FNC_TRIANGLE_CONTAIN

inline __host__ __device__ bool contain(const Triangle& _tri, const float3& _pos) { 
    // Move the triangle so that the point becomes the
    // triangles origin
    float3 local_a = _tri.a - _pos;
    float3 local_b = _tri.b - _pos;
    float3 local_c = _tri.c - _pos;

    // The point should be moved too, so they are both
    // relative, but because we don't use p in the
    // equation anymore, we don't need it!
    // p -= p;

    // Compute the normal vectors for triangles:
    // u = normal of PBC
    // v = normal of PCA
    // w = normal of PAB

    float3 u = cross(local_b, local_c);
    float3 v = cross(local_c, local_a);
    float3 w = cross(local_a, local_b);

    // Test to see if the normals are facing the same direction.
    // If yes, the point is inside, otherwise it isn't.
    return dot(u, v) >= 0.0f && dot(u, w) >= 0.0f;
}

#endif