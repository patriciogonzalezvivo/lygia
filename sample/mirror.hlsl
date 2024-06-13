#include "../sampler.hlsl"
#include "../math/mirror.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: fakes a mirror wrapping texture
use: <float4> sampleMirror(<SAMPLER_TYPE> tex, <float2> st);
options:
    - SAMPLER_FNC(TEX, UV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLEMIRROR
#define FNC_SAMPLEMIRROR
float4 sampleMirror(SAMPLER_TYPE tex, float2 st) { 
    return SAMPLER_FNC( tex, mirror(st) ); 
}
#endif