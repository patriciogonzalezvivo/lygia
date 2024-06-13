#include "cubic.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: cubic polynomial interpolation between two values
use: <float|vec2|vec3|vec4> cubicMix(<float|vec2|vec3|vec4> A, <float|vec2|vec3|vec4> B, float t)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

#ifndef FNC_CUBICMIX
#define FNC_CUBICMIX
float cubicMix(float A, float B, float t) { return A + (B - A) * cubic(t); }
vec2 cubicMix(vec2 A, vec2 B, float t) { return A + (B - A) * cubic(t); }
vec2 cubicMix(vec2 A, vec2 B, vec2 t) { return A + (B - A) * cubic(t); }
vec3 cubicMix(vec3 A, vec3 B, float t) { return A + (B - A) * cubic(t); }
vec3 cubicMix(vec3 A, vec3 B, vec3 t) { return A + (B - A) * cubic(t); }
vec4 cubicMix(vec4 A, vec4 B, float t) { return A + (B - A) * cubic(t); }
vec4 cubicMix(vec4 A, vec4 B, vec4 t) { return A + (B - A) * cubic(t); }
#endif