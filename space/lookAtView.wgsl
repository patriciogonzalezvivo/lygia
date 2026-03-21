/*
contributors:  Shadi El Hajj
description: Create a look-at view matrix
use: <mat4> lookAtView(in <vec3> position, in <vec3> target, in <vec3> up)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#include "lookAt.wgsl"
#include "translate.wgsl"

fn lookAtView3(position: vec3f, target: vec3f, up: vec3f) -> mat4x4<f32> {
    let m = lookAt(position, target, up);
    return translate(m, position);
}

fn lookAtView3a(position: vec3f, target: vec3f, roll: f32) -> mat4x4<f32> {
    let m = lookAt(position, target, roll);
    return translate(m, position);
}

fn lookAtView3b(position: vec3f, lookAt: vec3f) -> mat4x4<f32> {
    return lookAtView(position, lookAt, vec3f(0.0, 1.0, 0.0));
}
