
#include "../sdf/circleSDF.hlsl"

#include "fill.hlsl"
#include "stroke.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Draw a circle filled or not.
use: circle(<float2> st, <float> size [, <float> width])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_CIRCLE
#define FNC_CIRCLE
float circle(float2 st, float size) {
    return fill(circleSDF(st), size);
}

float circle(float2 st, float size, float strokeWidth) {
    return stroke(circleSDF(st), size, strokeWidth);
}
#endif