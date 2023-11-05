
#include "../sdf/triSDF.hlsl"

#include "fill.hlsl"
#include "stroke.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: draw a triangle filled or not. 
use: tri(<float2> st, <float> size [, <float> width])
*/

#ifndef FNC_TRI
#define FNC_TRI
float tri(float2 st, float size) {
    return fill(triSDF(st), size);
}

float tri(float2 st, float size, float strokeWidth) {
    return stroke(triSDF(st), size, strokeWidth);
}
#endif