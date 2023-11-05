#include "../space/nearest.hlsl"
#include "../sample.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: fakes a nearest sample
use: <float4? sampleNearest(<SAMPLER_TYPE> tex, <float2> st, <float2> texResolution);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef FNC_SAMPLENEARERS
#define FNC_SAMPLENEARERS
float4 sampleNearest(SAMPLER_TYPE tex, float2 st, float2 texResolution) {
    return SAMPLER_FNC( tex, nearest(st, texResolution) );
}
#endif