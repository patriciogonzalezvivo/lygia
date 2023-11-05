#include "../sample.hlsl"
#include "../math/mirror.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: fakes a mirror wrapping texture 
use: <float4> sampleMirror(<SAMPLER_TYPE> tex, <float2> st);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef FNC_SAMPLEMIRROR
#define FNC_SAMPLEMIRROR
float4 sampleMirror(SAMPLER_TYPE tex, float2 st) { 
    return SAMPLER_FNC( tex, mirror(st) ); 
}
#endif