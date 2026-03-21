#include "digits.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Draws all the digits of a matrix, useful for debugging.
use: <vec4> matrix(<vec2> st, <mat2|mat3|mat4> M)
options:
    DIGITS_DECIMALS: number of decimals after the point, defaults to 2
    DIGITS_SIZE: size of the font, defaults to vec2(.025)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn matrix2(st: vec2f, M: mat2x2<f32>) -> vec4f {
    let rta = vec4f(0.0);
    let size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * 2.0;
    rta.a = 0.5 * step(-DIGITS_SIZE.x, st.x) * step(st.x, size.x) * step(-DIGITS_SIZE.y, st.y) * step(st.y, size.y);
    rta += digits(st, M);
    return rta;
}

fn matrix2a(st: vec2f, M: mat3x3<f32>) -> vec4f {
    let rta = vec4f(0.0);
    let size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * 3.0;
    rta.a = 0.5 * step(-DIGITS_SIZE.x, st.x) * step(st.x, size.x) * step(-DIGITS_SIZE.y, st.y) * step(st.y, size.y);
    rta += digits(st, M);
    return rta;
}

fn matrix2b(st: vec2f, M: mat4x4<f32>) -> vec4f {
    let rta = vec4f(0.0);
    let size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * 4.0;
    rta.a = 0.5 * step(-DIGITS_SIZE.x, st.x) * step(st.x, size.x) * step(-DIGITS_SIZE.y, st.y) * step(st.y, size.y);
    rta += digits(st, M);
    return rta;
}
