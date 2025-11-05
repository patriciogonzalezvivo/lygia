@if(CENTER_3D)
import constants::CENTER_3D;

@if(CENTER_4D)
import constants::CENTER_4D;

import lygia::math::rotate4dX::rotate4dX;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Rotate a 3D/4D space around the X axis by a radian angle

    WESL conditionals:
    - @if(CENTER_3D): Uses custom 3D center point for vec3 rotations
    - @if(CENTER_4D): Uses custom 4D center point for vec4 rotations
use: |
    rotateX3(v: vec3f, r: f32) -> vec3f
    rotateX3_c(v: vec3f, r: f32, c: vec3f) -> vec3f
    rotateX4(v: vec4f, r: f32) -> vec4f
    rotateX4_c(v: vec4f, r: f32, c: vec4f) -> vec4f
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rotateX4_c(v: vec4f, r: f32, c: vec4f) -> vec4f {
    return rotate4dX(r) * (v - c) + c;
}

fn rotateX4(v: vec4f, r: f32) -> vec4f {
    @if(CENTER_4D)
    let centered = v - constants::CENTER_4D;
    @else
    let centered = v;

    let result = rotate4dX(r) * centered;

    @if(CENTER_4D)
    return result + constants::CENTER_4D;
    @else
    return result;
}

fn rotateX3_c(v: vec3f, r: f32, c: vec3f) -> vec3f {
    return (rotate4dX(r) * vec4f(v - c, 1.0)).xyz + c;
}

fn rotateX3(v: vec3f, r: f32) -> vec3f {
    @if(CENTER_3D)
    let centered = v - constants::CENTER_3D;
    @else
    let centered = v;

    let result = (rotate4dX(r) * vec4f(centered, 1.0)).xyz;

    @if(CENTER_3D)
    return result + constants::CENTER_3D;
    @else
    return result;
}
