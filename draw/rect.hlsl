
#include "../sdf/rectSDF.hlsl"

#include "fill.hlsl"
#include "stroke.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: draw a rectangel filled or not. 
use: rect(<float2> st, <float2> size [, <float> width])
*/

#ifndef FNC_RECT
#define FNC_RECT
float rect(float2 st, float2 size) {
    return fill(rectSDF(st, size), 1.0);
}

float rect(float2 st, float2 size, float strokeWidth) {
    return stroke(rectSDF(st, size), 1.0, strokeWidth);
}
#endif