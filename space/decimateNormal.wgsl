#include "../math/decimate.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Decimate normal vector by reducing the number of unique directions based on a given precision value.
use: <vec3> decimateNormal( <vec3> normal, <float> press )
license:
    - Copyright (c) 2025 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2025 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn decimateNormal(normal: vec3f, press: f32) -> vec3f {
    // convert normals into phi, theta angles
    let phi = atan(normal.y, normal.x);
    let theta = acos(normal.z);
    phi = decimate(phi, press);
    theta = decimate(theta, press);
    // convert angles into normal
    normal = vec3f(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    return normal;
}
