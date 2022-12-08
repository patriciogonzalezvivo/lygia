#include "saturate.glsl"
#include "quintic.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: quintic polynomial step function 
use: smoothstep(<float> in, <float> out, <float> value)
*/

#ifndef FNC_SMOOTHERSTEP
#define FNC_SMOOTHERSTEP
float smootherstep(float edge0, float edge1, float x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }
vec2  smootherstep(vec2  edge0, vec2  edge1, vec2  x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }
vec3  smootherstep(vec3  edge0, vec3  edge1, vec3  x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }
vec4  smootherstep(vec4  edge0, vec4  edge1, vec4  x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }
#endif