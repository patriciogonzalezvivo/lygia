#include "shadowLerp.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample shadow map using PCF
use:
    - <float> sampleShadowPCF(<SAMPLER_TYPE> depths, <float2> size, <float2> uv, <float> compare)
    - <float> sampleShadowPCF(<float3> lightcoord)
options:
    - LIGHT_SHADOWMAP_BIAS
    - SAMPLESHADOWPCF_SAMPLER_FNC
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SAMPLESHADOWPCF_SAMPLER_FNC
#define SAMPLESHADOWPCF_SAMPLER_FNC sampleShadowLerp
#endif

#ifndef FNC_SAMPLESHADOWPCF
#define FNC_SAMPLESHADOWPCF

float sampleShadowPCF(SAMPLER_TYPE depths, float2 size, float2 uv, float compare) {
    float2 pixel = 1.0/size;
    float result = 0.0;
    for (float x= -2.0; x <= 2.0; x++)
        for (float y= -2.0; y <= 2.0; y++) 
            result += SAMPLESHADOWPCF_SAMPLER_FNC(depths, size, uv + float2(x,y) * pixel, compare);
    return result/25.0;
}

#endif