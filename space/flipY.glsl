/*
original_author: Patricio Gonzalez Vivo
description: Flip Y axis
use: <vec2|vec3|vec4> flipY(<vec2|vec3|vec4> st)
*/

#ifndef FNC_FLIPY
#define FNC_FLIPY
vec2 flipY(in vec2 st) { return vec2(st.x, 1. - st.y); }
vec3 flipY(in vec3 st) { return vec3(st.x, 1. - st.y, st.z); }
vec4 flipY(in vec4 st) { return vec4(st.x, 1. - st.y, st.z, st.w); }
#endif
