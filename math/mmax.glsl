/*
original_author: Patricio Gonzalez Vivo
description: extend GLSL Max function to add more arguments
use: 
    - max(<float> A, <float> B, <float> C[, <float> D])
    - max(<vec2|vec3|vec4> A)
*/

#ifndef FNC_MMAX
#define FNC_MMAX

float mmax(in float a, in float b) { return max(a, b); }
float mmax(in float a, in float b, in float c) { return max(a, max(b, c)); }
float mmax(in float a, in float b, in float c, in float d) { return max(max(a, b), max(c, d)); }
float mmax(const vec2 v) { return max(v.x, v.y); }
float mmax(const vec3 v) { return mmax(v.x, v.y, v.z); }
float mmax(const vec4 v) { return mmax(v.x, v.y, v.z, v.w); }

#endif
