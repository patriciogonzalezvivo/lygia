/*
contributors: Inigo Quiles
description: inverse cubic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|vec2|vec3|vec4> invCubic(<float|vec2|vec3|vec4> value);
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/math_functions.frag
*/

#ifndef FNC_INVCUBIC
#define FNC_INVCUBIC 
float invCubic(const in float v) { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
vec2  invCubic(const in vec2 v)  { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
vec3  invCubic(const in vec3 v)  { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
vec4  invCubic(const in vec4 v)  { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
#endif