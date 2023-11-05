#include "saturate.glsl"
#include "quintic.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: quintic polynomial step function 
use: smoothstep(<float> in, <float> out, <float> value)
*/

#ifndef FNC_SMOOTHERSTEP
#define FNC_SMOOTHERSTEP
float smootherstep(float a, float b, float v) { return quintic( saturate( (v - a)/(b - a) )); }
vec2  smootherstep(vec2  a, vec2  b, vec2  v) { return quintic( saturate( (v - a)/(b - a) )); }
vec3  smootherstep(vec3  a, vec3  b, vec3  v) { return quintic( saturate( (v - a)/(b - a) )); }
vec4  smootherstep(vec4  a, vec4  b, vec4  v) { return quintic( saturate( (v - a)/(b - a) )); }
#endif