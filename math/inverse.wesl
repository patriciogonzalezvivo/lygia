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
