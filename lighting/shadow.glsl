#include "../sample/shadowPCF.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Sample shadow map using PCF
use:
    - <float> sampleShadowPCF(<SAMPLER_TYPE> depths, <vec2> size, <vec2> uv, <float> compare)
    - <float> sampleShadowPCF(<vec3> lightcoord)
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

float shadow(SAMPLER_TYPE shadoMap, const in vec2 size, const in vec2 uv, float compare) {
    #ifdef SHADOWMAP_BIAS
    compare -= SHADOWMAP_BIAS;
    #endif

    #if defined(PLATFORM_RPI) 
    return sampleShadow(shadoMap, size, uv, compare);
    #elif defined(TARGET_MOBILE)
    return sampleShadowLerp(shadoMap, size, uv, compare);
    #else 
    return sampleShadowPCF(shadoMap, size, uv, compare);
    #endif
}

#endif 