/*
contributors: Patricio Gonzalez Vivo
description: create a look at matrix. Right handed by default.
use:
    - <mat3> lookAt(<vec3> forward, <vec3> up)
    - <mat3> lookAt(<vec3> eye, <vec3> target, <vec3> up)
    - <mat3> lookAt(<vec3> eye, <vec3> target, <float> roll)
    - <mat3> lookAt(<vec3> forward)
options:
    - LOOK_AT_LEFT_HANDED: assume a left-handed coordinate system
    - LOOK_AT_RIGHT_HANDED: assume a right-handed coordinate system
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn lookAt3(forward: vec3f, up: vec3f) -> mat3x3<f32> {
    let zaxis = normalize(forward);
    let xaxis = normalize(cross(zaxis, up));
    let yaxis = cross(xaxis, zaxis);
    let xaxis = normalize(cross(up, zaxis));
    let yaxis = cross(zaxis, xaxis);
    return mat3x3<f32>(xaxis, yaxis, zaxis);
}

fn lookAt3a(eye: vec3f, target: vec3f, up: vec3f) -> mat3x3<f32> {
    let forward = normalize(target - eye);
    return lookAt(forward, up);
}

fn lookAt3b(eye: vec3f, target: vec3f, roll: f32) -> mat3x3<f32> {
    let up = vec3f(sin(roll), cos(roll), 0.0);
    return lookAt(eye, target, up);
}

fn lookAt3c(forward: vec3f) -> mat3x3<f32> {
    return lookAt(forward, vec3f(0.0, 1.0, 0.0));
}
