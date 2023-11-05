/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns A when cond is true, and B otherwise. This is in part to bring a compatibility layer with WGSL 
use: <float|vec2|vec3|vec4> select(<float|vec2|vec3|vec4> A, <float|vec2|vec3|vec4> B, <bool> cond)
*/

#ifndef FNC_SELECT
#define FNC_SELECT
int select(int A, int B, bool cond) { return cond ? A : B; }
float select(float A, float B, bool cond) { return cond ? A : B; }
vec2 select(vec2 A, vec2 B, bool cond) { return cond ? A : B; }
vec3 select(vec3 A, vec3 B, bool cond) { return cond ? A : B; }
vec4 select(vec4 A, vec4 B, bool cond) { return cond ? A : B; }
#endif