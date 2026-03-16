/*
contributors: Patricio Gonzalez Vivo
description: given a 4x4 returns a 3x3
use: <mat4> toMat3(<mat3> m)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn toMat3(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(m);
    return mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz);
}
