#include "triangle.cuh"
#include "../../math/const.cuh"
#include "../../math/dot.cuh"
#include "../../math/cross.cuh"
#include "../../math/operations.cuh"

#include "../../lighting/ray.cuh"
/*
contributors: Inigo Quiles
description: Based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
use: <float> intersect(<Triangle> tri, <float3> rayOrigin, <float3> rayDir, out <float3> intersectionPoint)
*/

#ifndef FNC_TRIANGLE_INTERSECT
#define FNC_TRIANGLE_INTERSECT

inline __host__ __device__ float intersect(const Triangle& _tri, const float3& _rayOrigin, const float3& _rayDir, float3& _point) {
    float3 v1v0 = _tri.b - _tri.a;
    float3 v2v0 = _tri.c - _tri.a;
    float3 rov0 = _rayOrigin - _tri.a;
    _point = cross(v1v0, v2v0);
    float3 q = cross(rov0, _rayDir);
    float d = 1.0f / dot(_rayDir, _point);
    float u = d * -dot(q, v2v0);
    float v = d *  dot(q, v1v0);
    float t = d * -dot(_point, rov0);
    if (u < 0.0f || u > 1.0f || v < 0.0f || (u+v) > 1.0f || t < 0.0f)
        t = MAX_FLOAT_VALUE; // No intersection

    return t;
}

inline __host__ __device__ float intersect(const Triangle& _tri, const Ray& _ray, float3& _point) { return intersect(_tri, _ray.origin, _ray.direction, _point); }
inline __host__ __device__ float intersect(const Triangle& _tri, const Ray& _ray) { 
    float3 p;
    return intersect(_tri, _ray, p); 
}

#endif