/*
contributors: Patricio Gonzalez Vivo
description: power of 7
use: <float|vec2|vec3|vec4> pow7(<float|vec2|vec3|vec4> v)
*/

#ifndef FNC_POW2
#define FNC_POW2

float pow7(const in float v) { return v * v * v * v * v * v * v; }
vec2 pow7(const in vec2 v) { return v * v * v * v * v * v * v; }
vec3 pow7(const in vec3 v) { return v * v * v * v * v * v * v; }
vec4 pow7(const in vec4 v) { return v * v * v * v * v * v * v; }

#endif
