#include "../aabb.cuh"
#include "../../math/make.cuh"
#include "../../math/min.cuh"
#include "../../math/max.cuh"
#include "../../math/operations.cuh"
#include "../../lighting/ray.cuh"

/*
original_author: Dominik Schmid 
description: compute the near and far intersections of the cube (stored in the x and y components) using the slab method
// no intersection means vec.x > vec.y (really tNear > tFar) https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
use: <float2> AABBintersect(<AABB> box, <float3> rayOrigin, <float3> rayDir)
*/

#ifndef FNC_AABB_INTERSECT
#define FNC_AABB_INTERSECT

inline __host__ __device__ float2 AABBintersect(const AABB& box, const float3& rayOrigin, const float3& rayDir) {
    float3 tMin = (box.min - rayOrigin) / rayDir;
    float3 tMax = (box.max - rayOrigin) / rayDir;
    float3 t1 = min(tMin, tMax);
    float3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return make_float2(tNear, tFar);
}

inline __host__ __device__ float2 AABBintersect(const AABB& box, const Ray& ray) {
    float3 tMin = (box.min - ray.origin) / ray.direction;
    float3 tMax = (box.max - ray.origin) / ray.direction;
    float3 t1 = min(tMin, tMax);
    float3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return make_float2(tNear, tFar);
}

#endif