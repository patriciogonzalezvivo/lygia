#include "../sdf/lineSDF.wgsl"
#include "fill.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Draw a line between two points. The thickness of the line can be adjusted.
use: <float> line(<vec2> st, <vec2> a, <vec2> b, <float> thickness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn line(st: vec2f, a: vec2f, b: vec2f, thickness: f32) -> f32 {
    return fill(lineSDF(st, a, b), thickness);
}
