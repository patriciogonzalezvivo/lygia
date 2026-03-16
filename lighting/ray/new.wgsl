#include "../ray.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Creates a new Ray assigning origin and direction
use:
    - rayNew(inout <Ray> ray, [<vec3> origin, <vec3> direction])
    - <Ray> rayNew(<vec3> origin, [<vec3> direction])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rayNew(ray: Ray, origin: vec3f, direction: vec3f) {
    ray.origin = origin;
    ray.direction = direction;
}

fn rayNewa(ray: Ray, origin: vec3f) {
    rayNew(ray, origin, vec3f(0.0, 0.0, 1.0));
}

fn rayNewb(ray: Ray) {
    rayNew(ray, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0));
}

Ray rayNew(vec3 origin, vec3 direction) {
    Ray ray;
    ray.origin = vec3f(0.0, 0.0, 0.0);
    ray.direction = vec3f(0.0, 0.0, 1.0);
    rayNew(ray, origin, direction);
    return ray;
}

Ray rayNew(vec3 origin) {
    Ray ray;
    ray.origin = vec3f(0.0, 0.0, 0.0);
    ray.direction = vec3f(0.0, 0.0, 1.0);
    rayNew(ray, origin);
    return ray;
}

Ray rayNew() {
    Ray ray;
    ray.origin = vec3f(0.0, 0.0, 0.0);
    ray.direction = vec3f(0.0, 0.0, 1.0);
    rayNew(ray);
    return ray;
}
