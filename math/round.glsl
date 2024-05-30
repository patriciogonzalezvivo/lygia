/*
contributors: Patricio Gonzalez Vivo
description: round a value to the nearest integer
use: <float|vec2|vec3|vec4> round(<float|vec2|vec3|vec4> value)
*/

#ifndef FNC_ROUND
#define FNC_ROUND
float round(float x) { return sign(x)*floor(abs(x)+0.5); }
vec2 round(vec2 x) { return sign(x)*floor(abs(x)+0.5); }
vec3 round(vec3 x) { return sign(x)*floor(abs(x)+0.5); }
vec4 round(vec4 x) { return sign(x)*floor(abs(x)+0.5); }
#endif