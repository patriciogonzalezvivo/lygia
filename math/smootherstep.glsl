#include "saturate.glsl"
#include "quintic.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: quintic polynomial step function
use: <float|vec2|vec3|vec4> smoothstep(<float|vec2|vec3|vec4> in, <float|vec2|vec3|vec4> out, <float|vec2|vec3|vec4> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SMOOTHERSTEP
#define FNC_SMOOTHERSTEP
float smootherstep(float a, float b, float v) { return quintic( saturate( (v - a)/(b - a) )); }
vec2  smootherstep(vec2  a, vec2  b, vec2  v) { return quintic( saturate( (v - a)/(b - a) )); }
vec3  smootherstep(vec3  a, vec3  b, vec3  v) { return quintic( saturate( (v - a)/(b - a) )); }
vec4  smootherstep(vec4  a, vec4  b, vec4  v) { return quintic( saturate( (v - a)/(b - a) )); }
#endif