/*
original_author: Inigo Quiles
description: inverse quartic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|vec2|vec3|vec4> invQuartic(<float|vec2|vec3|vec4> value);
*/

#ifndef FNC_INVQUARTIC
#define FNC_INVQUARTIC 

float invQuartic(const in float v) { return sqrt(1.0-sqrt(1.0-v)); }
vec2  invQuartic(const in vec2 v)  { return sqrt(1.0-sqrt(1.0-v)); }
vec3  invQuartic(const in vec3 v)  { return sqrt(1.0-sqrt(1.0-v)); }
vec4  invQuartic(const in vec4 v)  { return sqrt(1.0-sqrt(1.0-v)); }

#endif