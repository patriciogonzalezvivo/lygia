
#include "../sdf/crossSDF.hlsl"

#include "fill.hlsl"
#include "stroke.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: draw a cross filled or not. 
use: cross(<float2> st, <float> size [, <float> width])
*/

#ifndef FNC_CROSS
#define FNC_CROSS
float cross(float2 st, float size) {
    return fill(crossSDF(st, 1.), size);
}

float cross(float2 st, float size, float strokeWidth) {
    return stroke(crossSDF(st, 1.), size, strokeWidth);
}
#endif