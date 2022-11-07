#include "saturate.hlsl"
#include "quintic.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: quintic polynomial step function 
use: smoothstep(<float> in, <float> out, <float> value)
*/

#ifndef FNC_SMOOTHERSTEP
#define FNC_SMOOTHERSTEP
float smootherstep(float edge0, float edge1, float x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }
#endif