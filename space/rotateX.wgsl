#include "../math/rotate4dX.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateX(<vec3|vec4> v, float radian [, vec3 center])
options:
    - CENTER_3D
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rotateX4(v: vec4f, r: f32, c: vec4f) -> vec4f {
    return rotate4dX(r) * (v - c) + c;
}

fn rotateX4a(v: vec4f, r: f32) -> vec4f {
    return rotate4dX(r) * (v - CENTER_4D) + CENTER_4D;
    return rotate4dX(r) * v;
}

fn rotateX3(v: vec3f, r: f32, c: vec3f) -> vec3f {
    return (rotate4dX(r) * vec4f(v - c, 1.0)).xyz + c;
}

fn rotateX3a(v: vec3f, r: f32) -> vec3f {
    return (rotate4dX(r) * vec4f(v - CENTER_3D, 1.0)).xyz + CENTER_3D;
    return (rotate4dX(r) * vec4f(v, 1.0)).xyz;
}
