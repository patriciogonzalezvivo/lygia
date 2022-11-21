#include "../sample.hlsl"
#include "../math/mirror.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: fakes a mirror wrapping texture 
use: <float4> sampleMirror(<sampler2D> tex, <float2> st);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef FNC_SAMPLEMIRROR
#define FNC_SAMPLEMIRROR
float4 sampleMirror(sampler2D tex, float2 st) { 
    return SAMPLER_FNC( tex, mirror(st) ); 
}
#endif