#include "../sample/shadowPCF.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: sample shadow map using PCF
use:
    - <float> sampleShadowPCF(<sampler2D> depths, <vec2> size, <vec2> uv, <float> compare)
    - <float> sampleShadowPCF(<vec3> lightcoord)
options:
    - SHADOWMAP_BIAS
*/

#ifndef SHADOWMAP_BIAS
#define SHADOWMAP_BIAS 0.005
#endif

#ifndef SHADOW_SAMPLER_FNC
#define SHADOW_SAMPLER_FNC sampleShadowPCF
#endif

#ifndef FNC_SHADOW
#define FNC_SHADOW

float shadow(sampler2D shadoMap, const in vec2 size, const in vec2 uv, float compare) {
    #ifdef SHADOWMAP_BIAS
    compare -= SHADOWMAP_BIAS;
    #endif

    #if defined(PLATFORM_RPI) 
    return sampleShadow(shadoMap, size, uv, compare);
    #elif defined(TARGET_MOBILE) || defined(PLATFORM_WEBGL)
    return sampleShadowLerp(shadoMap, size, uv, compare);
    #else 
    return sampleShadowPCF(shadoMap, size, uv, compare);
    #endif
}

#endif 