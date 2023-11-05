
#include "../sdf/rectSDF.glsl"

#include "fill.glsl"
#include "stroke.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: draw a rectangel filled or not. 
use: rect(<vec2> st, <vec2> size [, <float> width])
*/

#ifndef FNC_RECT
#define FNC_RECT

float rect(vec2 st, vec2 size, float strokeWidth) {
    return stroke(rectSDF(st, size), 1.0, strokeWidth);
}

float rect(vec2 st, float size, float strokeWidth) {
    return stroke(rectSDF(st, vec2(size)), 1.0, strokeWidth);
}

float rect(vec2 st, vec2 size) {
    return fill(rectSDF(st, size), 1.0);
}

float rect(vec2 st, float size) {
    return fill(rectSDF(st, vec2(size)), 1.0);
}

#endif