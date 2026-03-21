#include "../sdf/triSDF.wgsl"

#include "fill.wgsl"
#include "stroke.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Draw a triangle filled or not.
use: tri(<vec2> st, <float> size [, <float> width])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn tri2(st: vec2f, size: f32) -> f32 {
    return fill(triSDF(st), size);
}

fn tri2a(st: vec2f, size: f32, strokeWidth: f32) -> f32 {
    return stroke(triSDF(st), size, strokeWidth);
}
