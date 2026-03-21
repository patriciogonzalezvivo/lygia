/*
contributors:  Shadi El Hajj
description: Add a translation component to a transform matrix
use: <mat4> translate(in <mat3> matrix, in <vec3> tranaslation)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

 fn translate(m: mat3x3<f32>, translation: vec3f) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4f(m[0], 0.0),
        vec4f(m[1], 0.0),
        vec4f(m[2], 0.0),
        vec4f(translation, 1.0)
    );
}
