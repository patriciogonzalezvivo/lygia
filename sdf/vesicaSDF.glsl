#include "circleSDF.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns an almond-shaped sdf
use: <float> vesicaSDF(<vec2> st, <float> w)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_VESICASDF
#define FNC_VESICASDF
float vesicaSDF(in vec2 st, in float w) {
    vec2 offset = vec2(w*0.5,0.);
    return max( circleSDF(st-offset),
                circleSDF(st+offset));
}

float vesicaSDF(in vec2 st) {
    return vesicaSDF(st, 0.5);
}
#endif
