#include ""
/*
original_author: Patricio Gonzalez Vivo  
description: 
use: <float|float3\float4> rgb2srgb(<float|float3|float4> srgb)
*/


#ifndef SRGB_GAMMA
#define SRGB_GAMMA 0.4545454545
#endif

#ifndef SRGB_ALPHA
#define SRGB_ALPHA 0.055
#endif

#ifndef FNC_RGB2SRGB
#define FNC_RGB2SRGB

float rgb2srgb(float channel) {
    if (channel <= 0.0031308)
        return 12.92 * channel;
    else
        return (1.0 + SRGB_ALPHA) * pow(channel, 0.4166666666666667) - SRGB_ALPHA;
}

float3 rgb2srgb(float3 rgb) {
    #if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) | defined(PLATFORM_WEBGL)
        return pow(rgb, float3(SRGB_GAMMA));
    #else 

        float3 rgb_lo = 12.92 * rgb;
        float3 rgb_hi = (1.0 + SRGB_ALPHA) * pow(rgb, float3(0.4166666666666667, 0.4166666666666667, 0.4166666666666667)) - SRGB_ALPHA;
        return lerp(rgb_lo, rgb_hi, step(float3(0.0031308, 0.0031308, 0.0031308), rgb));

    #endif
}

float4 rgb2srgb(float4 rgb) {
    return float4(rgb2srgb(rgb.rgb), rgb.a);
}

#endif