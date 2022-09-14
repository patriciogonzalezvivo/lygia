/*
original_author: Patricio Gonzalez Vivo
description: calculate diffuse contribution using lambert equation
use: 
    - <float> diffuseLambert(<vec3> light, <vec3> normal [, <vec3> view, <float> roughness] )
    - <float> diffuseLambert(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness)
*/

#ifndef FNC_DIFFUSE_LAMBERT
#define FNC_DIFFUSE_LAMBERT
float diffuseLambert(vec3 L, vec3 N) { return max(0.0, dot(N, L)); }
float diffuseLambert(vec3 L, vec3 N, vec3 V, float roughness) { return diffuseLambert(L, N); }
float diffuseLambert(vec3 L, vec3 N, vec3 V, float NoV, float NoL, float roughness) { return diffuseLambert(L, N); }
#endif