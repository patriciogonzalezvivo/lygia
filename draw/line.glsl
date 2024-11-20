#include "../sdf/lineSDF.glsl"
#include "fill.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Draw a line between two points. The thickness of the line can be adjusted.
use: <float> line(<vec2> st, <vec2> a, <vec2> b, <float> thickness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LINE
#define FNC_LINE

float line(vec2 st, vec2 a, vec2 b, float thickness) {
    return fill(lineSDF(st, a, b), thickness);
}

#endif