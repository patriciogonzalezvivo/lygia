/*
contributors: Mikola Lysenko
description: transpose matrixes
use: <mat3|mat4> transpose(in <mat3|mat4> m)
*/

fn transpose(m: mat3x3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(    m[0][0], m[1][0], m[2][0],
                    m[0][1], m[1][1], m[2][1],
                    m[0][2], m[1][2], m[2][2] );
}

fn transposea(m: mat4x4<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(    vec4f(m[0][0], m[1][0], m[2][0], m[3][0]),
                    vec4f(m[0][1], m[1][1], m[2][1], m[3][1]),
                    vec4f(m[0][2], m[1][2], m[2][2], m[3][2]),
                    vec4f(m[0][3], m[1][3], m[2][3], m[3][3])    );
}
