#include "../math/rotate4dY.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateY(<vec3> pos, float radian [, vec4 center])
options:
    - CENTER_3D
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rotateY4(v: vec4f, r: f32, c: vec4f) -> vec4f {
    return rotate4dY(r) * (v - c) + c;
}

fn rotateY4a(v: vec4f, r: f32) -> vec4f {
    return rotate4dY(r) * (v - CENTER_4D) + CENTER_4D;
    return rotate4dY(r) * v;
}

fn rotateY3(v: vec3f, r: f32, c: vec3f) -> vec3f {
    return (rotate4dY(r) * vec4f(v - c, 1.)).xyz + c;
}

fn rotateY3a(v: vec3f, r: f32) -> vec3f {
    return (rotate4dY(r) * vec4f(v - CENTER_3D, 1.)).xyz + CENTER_3D;
    return (rotate4dY(r) * vec4f(v, 1.)).xyz;
}
