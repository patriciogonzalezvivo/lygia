#include "../sample/shadowPCF.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Sample shadow map using PCF
use:
    - <float> sampleShadowPCF(<SAMPLER_TYPE> depths, <float2> size, <float2> uv, <float> compare)
    - <float> sampleShadowPCF(<float3> lightcoord)
options:
    - SHADOWMAP_BIAS
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SHADOWMAP_BIAS
#define SHADOWMAP_BIAS 0.005
#endif

#ifndef SHADOW_SAMPLER_FNC
#define SHADOW_SAMPLER_FNC sampleShadowPCF
#endif

#ifndef FNC_SHADOW
#define FNC_SHADOW

float shadow(SAMPLER_TYPE shadoMap, float2 size, float2 uv, float compare) {
    #ifdef SHADOWMAP_BIAS
    compare -= SHADOWMAP_BIAS;
    #endif
    return sampleShadowPCF(shadoMap, size, uv, compare);
}

#endif 