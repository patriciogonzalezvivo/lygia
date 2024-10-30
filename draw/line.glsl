#include "../sdf/lineSDF.glsl"
#include "fill.glsl"

#ifndef FNC_LINE
#define FNC_LINE

float line(vec2 st, vec2 a, vec2 b, float thickness) {
    return fill(lineSDF(st, a, b), thickness);
}

#endif