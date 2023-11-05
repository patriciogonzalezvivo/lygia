
#include "../sdf/circleSDF.glsl"

#include "fill.glsl"
#include "stroke.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: draw a circle filled or not. 
use: circle(<vec2> st, <float> size [, <float> width])
*/

#ifndef FNC_CIRCLE
#define FNC_CIRCLE
float circle(vec2 st, float size) {
    return fill(circleSDF(st), size);
}

float circle(vec2 st, float size, float strokeWidth) {
    return stroke(circleSDF(st), size, strokeWidth);
}
#endif