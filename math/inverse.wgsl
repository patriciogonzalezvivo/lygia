/*
contributors: Mikola Lysenko
description: inverse matrixes
*/

fn inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let a00 = m[0][0];
    let a01 = m[0][1];
    let a02 = m[0][2];
    let a10 = m[1][0];
    let a11 = m[1][1];
    let a12 = m[1][2];
    let a20 = m[2][0];
    let a21 = m[2][1];
    let a22 = m[2][2];

    let b01 = a22 * a11 - a12 * a21;
    let b11 = -a22 * a10 + a12 * a20;
    let b21 = a21 * a10 - a11 * a20;

    let det = a00 * b01 + a01 * b11 + a02 * b21;

    let A = vec3f(b01, (-a22 * a01 + a02 * a21), ( a12 * a01 - a02 * a11)) / det;
    let B = vec3f(b11, ( a22 * a00 - a02 * a20), (-a12 * a00 + a02 * a10)) / det;
    let C = vec3f(b21, (-a21 * a00 + a01 * a20), ( a11 * a00 - a01 * a10)) / det;

    return mat3x3<f32>(A, B, C);
}

fn inverse4(m: mat4x4<f32>) -> mat4x4<f32> {
    let a00 = m[0][0];
    let a01 = m[0][1];
    let a02 = m[0][2];
    let a03 = m[0][3];
    let a10 = m[1][0];
    let a11 = m[1][1];
    let a12 = m[1][2];
    let a13 = m[1][3];
    let a20 = m[2][0];
    let a21 = m[2][1];
    let a22 = m[2][2];
    let a23 = m[2][3];
    let a30 = m[3][0];
    let a31 = m[3][1];
    let a32 = m[3][2];
    let a33 = m[3][3];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

    return mat4x4<f32>(
        (a11 * b11 - a12 * b10 + a13 * b09) / det,
        (a02 * b10 - a01 * b11 - a03 * b09) / det,
        (a31 * b05 - a32 * b04 + a33 * b03) / det,
        (a22 * b04 - a21 * b05 - a23 * b03) / det,
        (a12 * b08 - a10 * b11 - a13 * b07) / det,
        (a00 * b11 - a02 * b08 + a03 * b07) / det,
        (a32 * b02 - a30 * b05 - a33 * b01) / det,
        (a20 * b05 - a22 * b02 + a23 * b01) / det,
        (a10 * b10 - a11 * b08 + a13 * b06) / det,
        (a01 * b08 - a00 * b10 - a03 * b06) / det,
        (a30 * b04 - a31 * b02 + a33 * b00) / det,
        (a21 * b02 - a20 * b04 - a23 * b00) / det,
        (a11 * b07 - a10 * b09 - a12 * b06) / det,
        (a00 * b09 - a01 * b07 + a02 * b06) / det,
        (a31 * b01 - a30 * b03 - a32 * b00) / det,
        (a20 * b03 - a21 * b01 + a22 * b00) / det
    );
}