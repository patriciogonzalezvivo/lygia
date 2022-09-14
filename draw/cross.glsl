
#include "../sdf/crossSDF.glsl"

#include "fill.glsl"
#include "stroke.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: draw a cross filled or not. 
use: cross(<vec2> st, <float> size [, <float> width])
*/

#ifndef FNC_CROSS
#define FNC_CROSS
float cross(vec2 st, float size) {
    return fill(crossSDF(st, 1.), size);
}

float cross(vec2 st, float size, float strokeWidth) {
    return stroke(crossSDF(st, 1.), size, strokeWidth);
}
#endif