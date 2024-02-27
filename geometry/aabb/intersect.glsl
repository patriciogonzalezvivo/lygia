#include "aabb.glsl"
#include "../../lighting/ray.glsl"

/*
contributors: Dominik Schmid 
description: |
    Compute the near and far intersections of the cube (stored in the x and y components) using the slab method
    no intersection means vec.x > vec.y (really tNear > tFar) https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
use: <vec2> intersect(<AABB> box, <vec3> rayOrigin, <vec3> rayDir)
*/

#ifndef FNC_AABB_INTERSECT
#define FNC_AABB_INTERSECT

vec2 intersect(const in AABB box, const in vec3 rayOrigin, const in vec3 rayDir) {
    vec3 tMin = (box.min - rayOrigin) / rayDir;
    vec3 tMax = (box.max - rayOrigin) / rayDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

vec2 intersect(const in AABB box, const in Ray ray) {
    vec3 tMin = (box.min - ray.origin) / ray.direction;
    vec3 tMax = (box.max - ray.origin) / ray.direction;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec2(tNear, tFar);
}

#endif