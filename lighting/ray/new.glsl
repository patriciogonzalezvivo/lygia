#include "../ray.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Creates a new Ray asigning origin and direction
use:
    - rayNew(inout <Ray> ray, [<vec3> origin, <vec3> direction])
    - <Ray> rayNew(<vec3> origin, [<vec3> direction])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RAYNEW
#define FNC_RAYNEW

void rayNew(inout Ray ray, vec3 origin, vec3 direction) {
    ray.origin = origin;
    ray.direction = direction;
}

void rayNew(inout Ray ray, vec3 origin) {
    rayNew(ray, origin, vec3(0.0, 0.0, 1.0));
}

void rayNew(inout Ray ray) {
    rayNew(ray, vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0));
}

Ray rayNew(vec3 origin, vec3 direction) {
    Ray ray;
    ray.origin = vec3(0.0, 0.0, 0.0);
    ray.direction = vec3(0.0, 0.0, 1.0);
    rayNew(ray, origin, direction);
    return ray;
}

Ray rayNew(vec3 origin) {
    Ray ray;
    ray.origin = vec3(0.0, 0.0, 0.0);
    ray.direction = vec3(0.0, 0.0, 1.0);
    rayNew(ray, origin);
    return ray;
}

Ray rayNew() {
    Ray ray;
    ray.origin = vec3(0.0, 0.0, 0.0);
    ray.direction = vec3(0.0, 0.0, 1.0);
    rayNew(ray);
    return ray;
}

#endif