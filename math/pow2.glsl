/*
contributors: Patricio Gonzalez Vivo
description: power of 2
use: <float|vec2|vec3|vec4> pow2(<float|vec2|vec3|vec4> v)
*/

#ifndef FNC_POW2
#define FNC_POW2

float pow2(const in float v) { return v * v; }
vec2 pow2(const in vec2 v) { return v * v; }
vec3 pow2(const in vec3 v) { return v * v; }
vec4 pow2(const in vec4 v) { return v * v; }

#endif
