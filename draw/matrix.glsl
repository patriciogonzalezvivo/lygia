#include "digits.glsl"

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

#ifndef FNC_DRAW_MATRIX
#define FNC_DRAW_MATRIX

vec4 matrix(in vec2 st, in mat2 M) {
    vec4 rta = vec4(0.0);
    vec2 size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * 2.0;
    rta.a = 0.5 * step(-DIGITS_SIZE.x, st.x) * step(st.x, size.x) * step(-DIGITS_SIZE.y, st.y) * step(st.y, size.y);
    rta += digits(st, M);
    return rta;
}

vec4 matrix(in vec2 st, in mat3 M) {
    vec4 rta = vec4(0.0);
    vec2 size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * 3.0;
    rta.a = 0.5 * step(-DIGITS_SIZE.x, st.x) * step(st.x, size.x) * step(-DIGITS_SIZE.y, st.y) * step(st.y, size.y);
    rta += digits(st, M);
    return rta;
}

vec4 matrix(in vec2 st, in mat4 M) {
    vec4 rta = vec4(0.0);
    vec2 size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * 4.0;
    rta.a = 0.5 * step(-DIGITS_SIZE.x, st.x) * step(st.x, size.x) * step(-DIGITS_SIZE.y, st.y) * step(st.y, size.y);
    rta += digits(st, M);
    return rta;
}

#endif