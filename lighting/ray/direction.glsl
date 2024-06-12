#include "../ray.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Set the direction (and origin in some cases) of a ray
use:
    - rayDirection(inout <Ray> ray, [<vec3> origin,] <vec2> pos, <vec2> resolution, <float> fov)
    - <Ray> rayDirection(<vec3> origin, <vec2> pos, <vec2> resolution, <float> fov)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RAYDIRECTION
#define FNC_RAYDIRECTION

void rayDirection(inout Ray ray, vec2 pos, vec2 resolution, float fov) {
    float aspect = resolution.x / resolution.y;
    float tanVFov2 = tan(fov / 2.0);
    float tanHFov2 = tanVFov2 * aspect;
    ray.direction = vec3(1.0,  0.0,  0.0) * (pos.x * tanHFov2) -
                    vec3(0.0, -1.0,  0.0) * (pos.y * tanVFov2) + 
                    vec3(0.0,  0.0, -1.0);
}

void rayDirection(inout Ray ray, vec3 origin, vec2 pos, vec2 resolution, float fov) {
    float aspect = resolution.x / resolution.y;
    float tanVFov2 = tan(fov / 2.0);
    float tanHFov2 = tanVFov2 * aspect;
    ray.origin = origin;
    ray.direction = vec3(1.0,  0.0,  0.0) * (pos.x * tanHFov2) -
                    vec3(0.0, -1.0,  0.0) * (pos.y * tanVFov2) + 
                    vec3(0.0,  0.0, -1.0);
}

Ray rayDirection(vec3 origin, vec2 pos, vec2 resolution, float fov) {
    Ray ray;
    ray.origin = vec3(0.0, 0.0, 0.0);
    ray.direction = vec3(0.0, 0.0, 1.0);
    rayDirection(ray, origin, pos, resolution, fov);
    return ray;
}

#endif