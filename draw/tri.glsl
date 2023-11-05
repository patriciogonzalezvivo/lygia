
#include "../sdf/triSDF.glsl"

#include "fill.glsl"
#include "stroke.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: draw a triangle filled or not. 
use: tri(<vec2> st, <float> size [, <float> width])
*/

#ifndef FNC_TRI
#define FNC_TRI
float tri(vec2 st, float size) {
    return fill(triSDF(st), size);
}

float tri(vec2 st, float size, float strokeWidth) {
    return stroke(triSDF(st), size, strokeWidth);
}
#endif