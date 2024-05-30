/*
contributors: Patricio Gonzalez Vivo
description: Squared length
use: <vec4|vec3|vec2> lengthSq(<vec4|vec3|vec2> v)
*/

#ifndef FNC_LENGTHSQ
#define FNC_LENGTHSQ

float lengthSq(in vec2 v) { return dot(v, v); }
float lengthSq(in vec3 v) { return dot(v, v); }
float lengthSq(in vec4 v) { return dot(v, v); }

#endif
