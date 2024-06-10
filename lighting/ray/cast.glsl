#include "../ray.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Cast ray by specific distance
use: <vec3> rayCast(<Ray> ray, <float> dist)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RAYCAST
#define FNC_RAYCAST

vec3 rayCast(Ray ray, float dist) {
    return ray.origin + ray.direction * dist;
}

#endif