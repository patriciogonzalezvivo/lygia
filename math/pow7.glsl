/*
original_author: Patricio Gonzalez Vivo
description: power of 7
use: pow7(<float|vec2|vec3|vec4> x)
*/

#ifndef FNC_POW2
#define FNC_POW2

float pow7(const in float x) { return x * x * x * x * x * x * x; }
vec2 pow7(const in vec2 x) { return x * x * x * x * x * x * x; }
vec3 pow7(const in vec3 x) { return x * x * x * x * x * x * x; }
vec4 pow7(const in vec4 x) { return x * x * x * x * x * x * x; }

#endif
