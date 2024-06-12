#include "../space/nearest.hlsl"
#include "../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: fakes a nearest sample
use: <float4? sampleNearest(<SAMPLER_TYPE> tex, <float2> st, <float2> texResolution);
options:
    - SAMPLER_FNC(TEX, UV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLENEARERS
#define FNC_SAMPLENEARERS
float4 sampleNearest(SAMPLER_TYPE tex, float2 st, float2 texResolution) {
    return SAMPLER_FNC( tex, nearest(st, texResolution) );
}
#endif