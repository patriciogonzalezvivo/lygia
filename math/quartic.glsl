/*
contributors: Inigo Quiles
description: quartic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|vec2|vec3|vec4> quartic(<float|vec2|vec3|vec4> value);
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

#ifndef FNC_QUARTIC
#define FNC_QUARTIC 

float quartic(const in float v) { return v*v*(2.0-v*v); }
vec2  quartic(const in vec2 v)  { return v*v*(2.0-v*v); }
vec3  quartic(const in vec3 v)  { return v*v*(2.0-v*v); }
vec4  quartic(const in vec4 v)  { return v*v*(2.0-v*v); }

#endif