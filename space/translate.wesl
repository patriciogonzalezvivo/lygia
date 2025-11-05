/*
contributors:  Shadi El Hajj
description: Add a translation component to a transform matrix
use: <mat4> translate(in <mat3> matrix, in <vec3> tranaslation)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn translate(m: mat3x3f, translation: vec3f) -> mat4x4f {
    return mat4x4f(
        vec4f(m[0], 0.0),
        vec4f(m[1], 0.0),
        vec4f(m[2], 0.0),
        vec4f(translation, 1.0)
    );
}
