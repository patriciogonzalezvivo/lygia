
#include "../sdf/circleSDF.hlsl"

#include "fill.hlsl"
#include "stroke.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: draw a circle filled or not. 
use: circle(<float2> st, <float> size [, <float> width])
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