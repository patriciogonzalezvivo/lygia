#include "../sdf/rectSDF.wgsl"

#include "fill.wgsl"
#include "stroke.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Draw a rectangel filled or not.
use: rect(<vec2> st, <vec2> size [, <float> width])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rect2(st: vec2f, size: vec2f, strokeWidth: f32) -> f32 {
    return stroke(rectSDF(st, size), 1.0, strokeWidth);
}

fn rect2a(st: vec2f, size: f32, strokeWidth: f32) -> f32 {
    return stroke(rectSDF(st, vec2f(size)), 1.0, strokeWidth);
}

fn rect2b(st: vec2f, size: vec2f) -> f32 {
    return fill(rectSDF(st, size), 1.0);
}

fn rect2c(st: vec2f, size: f32) -> f32 {
    return fill(rectSDF(st, vec2f(size)), 1.0);
}
