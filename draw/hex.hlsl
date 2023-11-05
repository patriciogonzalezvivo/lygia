
#include "../sdf/hexSDF.hlsl"

#include "fill.hlsl"
#include "stroke.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: draw a hexagon filled or not. 
use: hex(<float2> st, <float> size [, <float> width])
*/

#ifndef FNC_HEX
#define FNC_HEX
float hex(float2 st, float size) {
    return fill(hexSDF(st), size);
}

float hex(float2 st, float size, float strokeWidth) {
    return stroke(hexSDF(st), size, strokeWidth);
}
#endif