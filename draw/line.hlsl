#include "../sdf/lineSDF.hlsl"
#include "fill.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Draw a line between two points. The thickness of the line can be adjusted.
use: <float> line(<float2> st, <float2> a, <float2> b, <float> thickness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LINE
#define FNC_LINE

float line(float2 st, float2 a, float2 b, float thickness) {
    return fill(lineSDF(st, a, b), thickness);
}

#endif