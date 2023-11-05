/*
contributors: Patricio Gonzalez Vivo
description: |
    Moves the center from 0.0 to 0.5
use: <float|vec2|vec3> uncenter(<float|vec2|vec3> st)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_UNCENTER
#define FNC_UNCENTER

float uncenter(float v) { return v * 0.5 + 0.5; }
vec2  uncenter(vec2 v) { return v * 0.5 + 0.5; }
vec3  uncenter(vec3 v) { return v * 0.5 + 0.5; }

#endif