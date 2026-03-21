#include "aabb.wgsl"
#include "../../lighting/ray.wgsl"

/*
contributors: Dominik Schmid 
description: |
    Compute the near and far intersections of the cube (stored in the x and y components) using the slab method
    no intersection means vec.x > vec.y (really tNear > tFar) https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
use: <vec2> intersect(<AABB> box, <vec3> rayOrigin, <vec3> rayDir)
*/

fn intersect(box: AABB, rayOrigin: vec3f, rayDir: vec3f) -> vec2f {
    let tMin = (box.min - rayOrigin) / rayDir;
    let tMax = (box.max - rayOrigin) / rayDir;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);
    return vec2f(tNear, tFar);
}

fn intersecta(box: AABB, ray: Ray) -> vec2f {
    let tMin = (box.min - ray.origin) / ray.direction;
    let tMax = (box.max - ray.origin) / ray.direction;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);
    return vec2f(tNear, tFar);
}
