#include "digits.wgsl"
#include "circle.wgsl"

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

fn point2(st: vec2f, pos: vec2f, color: vec3f, radius: f32) -> vec4f {
    let rta = vec4f(0.0);
    let st_p = st - pos;

    rta += vec4f(color, 1.0) * circle(st_p + 0.5, radius);
    st_p -= DIGITS_SIZE * vec2f(0.0, 0.5);
    let size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * vec2f(2.0, 0.5);
    rta.a += 0.5 * step(0.0, st_p.x) * step(st_p.x, size.x) * step(-DIGITS_SIZE.y * 0.5, st_p.y) * step(st_p.y, size.y);
    rta += digits(st_p, pos);

    return rta;
}

fn point2a(st: vec2f, M: mat4x4<f32>, pos: vec3f, color: vec3f, radius: f32) -> vec4f {
    let rta = vec4f(0.0);
    let pos4 = M * vec4f(pos, 1.0);
    let p = pos4.xy / pos4.w;

    let st_p = st + (p.xy * 0.5 - 0.5);
    let st_p = st - (p.xy * 0.5 + 0.5);

    rta += vec4f(color, 1.0) * circle(st_p + 0.5, radius);
    st_p -= DIGITS_SIZE * vec2f(0.0, 0.5);
    let size = DIGITS_SIZE * abs(DIGITS_VALUE_OFFSET) * vec2f(3.0, 0.5);
    rta.a += 0.5 * step(0.0, st_p.x) * step(st_p.x, size.x) * step(-DIGITS_SIZE.y * 0.5, st_p.y) * step(st_p.y, size.y);
    rta += digits(st_p, pos);
    
    return rta;
}

fn point2b(st: vec2f, pos: vec2f) -> vec4f { return point(st, pos, vec3f(1.0, 0.0, 0.0), 0.02); }
fn point2c(st: vec2f, M: mat4x4<f32>, pos: vec3f) -> vec4f { return point(st, M, pos, vec3f(1.0, 0.0, 0.0), 0.02); }
