/*
original_author: Patricio Gonzalez Vivo  
description: 
use: <float|float3\float4> srgb2rgb(<float|float3|float4> srgb)
*/


#ifndef SRGB_INVERSE_GAMMA
#define SRGB_INVERSE_GAMMA 2.2
#endif

#ifndef SRGB_ALPHA
#define SRGB_ALPHA 0.055
#endif

#ifndef FNC_SRGB2RGB
#define FNC_SRGB2RGB

float srgb2rgb(float channel) {
    if (channel <= 0.04045)
        return channel * 0.08333333333; // 1. / 12.92;
    else
        return pow((channel + SRGB_ALPHA) / (1.0 + SRGB_ALPHA), 2.4);
}

float3 srgb2rgb(float3 srgb) {
    #if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) | defined(PLATFORM_WEBGL)
        return pow(srgb, float3(SRGB_INVERSE_GAMMA));
    #else 
        float3 srgb_lo = srgb / 12.92;
        float3 srgb_hi = pow((srgb + SRGB_ALPHA)/(1.0 + SRGB_ALPHA), float3(2.4, 2.4, 2.4));
        return mix(srgb_lo, srgb_hi, step(float3(0.04045, 0.04045, 0.04045), srgb));
    #endif
}

float4 srgb2rgb(float4 srgb) {
    return float4(srgb2rgb(srgb.rgb), srgb.a);
}

#endif