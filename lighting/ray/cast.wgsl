#include "../ray.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Cast ray by specific distance
use: <vec3> rayCast(<Ray> ray, <float> dist)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rayCast(ray: Ray, dist: f32) -> vec3f {
    return ray.origin + ray.direction * dist;
}
