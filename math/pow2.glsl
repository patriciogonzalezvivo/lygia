/*
original_author: Patricio Gonzalez Vivo
description: power of 2
use: pow2(<float|vec2|vec3|vec4> x)
*/

#ifndef FNC_POW2
#define FNC_POW2

float pow2(const in float x) { return x * x; }
vec2 pow2(const in vec2 x) { return x * x; }
vec3 pow2(const in vec3 x) { return x * x; }
vec4 pow2(const in vec4 x) { return x * x; }

#endif
