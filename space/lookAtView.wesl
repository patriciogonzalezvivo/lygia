import lygia::space::lookAt::{lookAt, lookAt3, lookAt3Roll};

/*
contributors: Shadi El Hajj
description: Create a look-at view matrix
use: <mat4x4f> lookAtView(<vec3f> position, <vec3f> center, <vec3f> up)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn lookAtView(position: vec3f, center: vec3f, up: vec3f) -> mat4x4f {
    let m3 = lookAt3(position, center, up);
    let m4 = mat4x4f(
        vec4f(m3[0], 0.0),
        vec4f(m3[1], 0.0),
        vec4f(m3[2], 0.0),
        vec4f(position, 1.0)
    );
    return m4;
}

fn lookAtViewRoll(position: vec3f, center: vec3f, roll: f32) -> mat4x4f {
    let m3 = lookAt3Roll(position, center, roll);
    let m4 = mat4x4f(
        vec4f(m3[0], 0.0),
        vec4f(m3[1], 0.0),
        vec4f(m3[2], 0.0),
        vec4f(position, 1.0)
    );
    return m4;
}

fn lookAtViewFromDirection(position: vec3f, direction: vec3f) -> mat4x4f {
    return lookAtView(position, direction, vec3f(0.0, 1.0, 0.0));
}