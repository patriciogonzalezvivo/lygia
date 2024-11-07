
#include "digits.glsl"
#include "circle.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Draws all the digits of a matrix, useful for debugging.
use:
    - <vec4> point(<vec2> st, <vec2> pos)
    - <vec4> point(<vec2> st, <mat4> P, <mat4> V, <vec3> pos)
options:
    DIGITS_DECIMALS: number of decimals after the point, defaults to 2
    DIGITS_SIZE: size of the font, defaults to vec2(.025)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_DRAW_POINT
#define FNC_DRAW_POINT

vec4 point(in vec2 st, vec2 pos, vec3 color, float radius) {
    vec4 rta = vec4(0.0);
    vec2 st_p = st - pos;

    rta += vec4(color, 1.0) * circle(st_p + 0.5, radius);
    st_p -= DIGITS_SIZE * vec2(0.0, 0.5);
    vec2 size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * vec2(2.0, 0.5);
    rta.a += 0.5 * step(0.0, st_p.x) * step(st_p.x, size.x) * step(-DIGITS_SIZE.y * 0.5, st_p.y) * step(st_p.y, size.y);
    rta += digits(st_p, pos);

    return rta;
}

vec4 point(in vec2 st, mat4 M, vec3 pos, vec3 color, float radius) {
    vec4 rta = vec4(0.0);
    vec4 pos4 = M * vec4(pos, 1.0);
    vec2 p = pos4.xy / pos4.w;

    #ifdef DEBUG_FLIPPED_SPACE
    vec2 st_p = st + (p.xy * 0.5 - 0.5);
    #else
    vec2 st_p = st - (p.xy * 0.5 + 0.5);
    #endif

    rta += vec4(color, 1.0) * circle(st_p + 0.5, radius);
    st_p -= DIGITS_SIZE * vec2(0.0, 0.5);
    vec2 size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * vec2(3.0, 0.5);
    rta.a += 0.5 * step(0.0, st_p.x) * step(st_p.x, size.x) * step(-DIGITS_SIZE.y * 0.5, st_p.y) * step(st_p.y, size.y);
    rta += digits(st_p, pos);
    
    return rta;
}

vec4 point(in vec2 st, vec2 pos) { return point(st, pos, vec3(1.0, 0.0, 0.0), 0.02); }
vec4 point(in vec2 st, mat4 M, vec3 pos) { return point(st, M, pos, vec3(1.0, 0.0, 0.0), 0.02); }

#endif