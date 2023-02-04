/*
original_author: Patricio Gonzalez Vivo
description: power of 3
use: pow3(<float|vec2|vec3|vec4> x)
*/

#ifndef FNC_POW3
#define FNC_POW3

float pow3(const in float x) { return x * x * x; }
vec2 pow3(const in vec2 x) { return x * x * x; }
vec3 pow3(const in vec3 x) { return x * x * x; }
vec4 pow3(const in vec4 x) { return x * x * x; }

#endif
