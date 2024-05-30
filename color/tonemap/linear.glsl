/*
contributors: nan
description: Linear tonemap (no modifications are applied)
use: <vec3|vec4> tonemapLinear(<vec3|vec4> x)
*/

#ifndef FNC_TONEMAPLINEAR
#define FNC_TONEMAPLINEAR
vec3 tonemapLinear(const vec3 v) { return v; }
vec4 tonemapLinear(const vec4 v) { return v; }
#endif