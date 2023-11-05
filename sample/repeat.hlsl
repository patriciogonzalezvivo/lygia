#include "../sample.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: fakes a repeat wrapping texture 
use: <float4> sampleRepeat(<SAMPLER_TYPE> tex, <float2> st);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef FNC_SAMPLEREPEAT
#define FNC_SAMPLEREPEAT
float4 sampleRepeat(SAMPLER_TYPE tex, float2 st) { 
    return SAMPLER_FNC( tex, frac(st) ); 
}
#endif