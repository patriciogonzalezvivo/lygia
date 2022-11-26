/*
original_author: Patricio Gonzalez Vivo
description: bump in a range between -1 and 1
use: <float> bump(<float> x, <float> k)
*/

#ifndef FNC_BUMP
#define FNC_BUMP
float bump(float x, float k){ return saturate( (1.0 - x * x) - k); }
float3 bump(float3 x, float3 k){ return saturate( (1.0 - x * x) - k); }
#endif