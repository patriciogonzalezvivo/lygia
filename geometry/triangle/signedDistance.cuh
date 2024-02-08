#include "triangle.cuh"
#include "normal.cuh"
#include "closestPoint.cuh"
#include "../../math/length.cuh"
#include "../../math/sign.cuh"
#include "../../math/operations.cuh"

/*
contributors: 
description: Returns the signed distance from the surface of a triangle to a point
use: <float3> closestDistance(<Triangle> tri, <float3> _pos) 
*/

#ifndef FNC_TRIANGLE_SIGNED_DISTANCE
#define FNC_TRIANGLE_SIGNED_DISTANCE

inline __host__ __device__ float signedDistance(const Triangle& _tri, const float3& _triNormal, const float3& _p) {
    float3 nearest = closestPoint(_tri, _triNormal, _p);
    float3 delta = _p - nearest;
    float distance = length(delta);
    distance *= sign( dot(delta/distance, _triNormal) );
    return distance;
}

inline __host__ __device__ float signedDistance(const Triangle& _tri, const float3& _p) { return signedDistance(_tri, normal(_tri), _p); }

#endif