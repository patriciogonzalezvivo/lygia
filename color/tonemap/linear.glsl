/*
Author: nan
description: linear tonempa (no modifications are applied)
use: <vec3|vec4> tonemapLinear(<vec3|vec4> x)
*/

#ifndef FNC_TONEMAPLINEAR
#define FNC_TONEMAPLINEAR
vec3 tonemapLinear(const vec3 x) { return x; }
vec4 tonemapLinear(const vec4 x) { return x; }
#endif