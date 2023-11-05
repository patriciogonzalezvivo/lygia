
#include "../sdf/hexSDF.glsl"

#include "fill.glsl"
#include "stroke.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: draw a hexagon filled or not. 
use: hex(<vec2> st, <float> size [, <float> width])
*/

#ifndef FNC_HEX
#define FNC_HEX
float hex(vec2 st, float size) {
    return fill(hexSDF(st), size);
}

float hex(vec2 st, float size, float strokeWidth) {
    return stroke(hexSDF(st), size, strokeWidth);
}
#endif