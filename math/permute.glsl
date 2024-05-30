#include "mod289.glsl"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description: permute
use: <float|vec2|vec3|vec4> permute(<float|vec2|vec3|vec4> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

#ifndef FNC_PERMUTE
#define FNC_PERMUTE

float permute(const in float v) { return mod289(((v * 34.0) + 1.0) * v); }
vec2 permute(const in vec2 v) { return mod289(((v * 34.0) + 1.0) * v); }
vec3 permute(const in vec3 v) { return mod289(((v * 34.0) + 1.0) * v); }
vec4 permute(const in vec4 v) { return mod289(((v * 34.0) + 1.0) * v); }

#endif
