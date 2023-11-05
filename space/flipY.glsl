/*
contributors: Patricio Gonzalez Vivo
description: Flip Y axis
use: <vec2|vec3|vec4> flipY(<vec2|vec3|vec4> st)
*/

#ifndef FNC_FLIPY
#define FNC_FLIPY
vec2 flipY(in vec2 v) { return vec2(v.x, 1. - v.y); }
vec3 flipY(in vec3 v) { return vec3(v.x, 1. - v.y, v.z); }
vec4 flipY(in vec4 v) { return vec4(v.x, 1. - v.y, v.z, v.w); }
#endif
