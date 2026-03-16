#include "../sdf/circleSDF.wgsl"

#include "fill.wgsl"
#include "stroke.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Draw a circle filled or not.
use: <float> circle(<vec2> st, <float> size [, <float> width])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn circle2(st: vec2f, size: f32) -> f32 {
    return fill(circleSDF(st), size);
}

fn circle2a(st: vec2f, size: f32, strokeWidth: f32) -> f32 {
    return stroke(circleSDF(st), size, strokeWidth);
}
