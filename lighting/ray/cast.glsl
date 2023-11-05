#include "../ray.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: cast ray by specific distance
use: <vec3> rayCast(<Ray> ray, <float> dist)
*/

#ifndef FNC_RAYCAST
#define FNC_RAYCAST

vec3 rayCast(Ray ray, float dist) {
    return ray.origin + ray.direction * dist;
}

#endif