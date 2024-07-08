#include "../../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: blue noise dithering
use:
    - <float4|float3|float> ditherBlueNoise(<float4|float3|float> value, <float2> fragcoord, <float> time)
    - <float4|float3|float> ditherBlueNoise(<float4|float3|float> value, <float> time)
options:
    - SAMPLER_FNC
    - BLUENOISE_TEXTURE
    - BLUENOISE_TEXTURE_RESOLUTION
    - DITHER_BLUENOISE_CHROMATIC
    - DITHER_BLUENOISE_ANIMATED
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef BLUENOISE_TEXTURE_RESOLUTION
#define BLUENOISE_TEXTURE_RESOLUTION float2(1024.0, 1024.0)
#endif

#ifdef DITHER_ANIMATED
#define DITHER_BLUENOISE_ANIMATED
#endif

#ifdef DITHER_CHROMATIC
#define DITHER_BLUENOISE_CHROMATIC
#endif

#ifndef FNC_DITHER_BLUENOISE
#define FNC_DITHER_BLUENOISE

float remap_pdf_tri_unity( float v ) {
    v = v*2.0-1.0;
    v = sign(v) * (1.0 - sqrt(1.0 - abs(v)));
    return 0.5 + 0.5*v;
}

static const float2 blueNoiseTexturePixel = 1.0/BLUENOISE_TEXTURE_RESOLUTION;

float ditherBlueNoise(SAMPLER_TYPE tex, in float b, float2 fragcoord, const in float time) {
    #ifdef DITHER_BLUENOISE_ANIMATED 
    fragcoord += 1337.0 * frac(time);
    #endif
    float bn = SAMPLER_FNC(tex, fragcoord * blueNoiseTexturePixel).r;
    float bn_tri = remap_pdf_tri_unity(bn);
    return b + (bn_tri*2.0-0.5)/255.0;
}

float3 ditherBlueNoise(SAMPLER_TYPE tex, in float3 rgb, float2 fragcoord, const in float time) {
    #ifdef DITHER_BLUENOISE_ANIMATED
    fragcoord += 1337.0 * frac(time * 0.1);
    #endif
        
    #ifdef DITHER_BLUENOISE_CHROMATIC
    float3 bn = SAMPLER_FNC(tex, fragcoord * blueNoiseTexturePixel).rgb;
    float3 bn_tri = float3( remap_noise_tri_erp(bn.x), 
                        remap_noise_tri_erp(bn.y), 
                        remap_noise_tri_erp(bn.z) );
    rgb += (bn_tri*2.0-0.5)/255.0;
    #else
    float bn = SAMPLER_FNC(tex, fragcoord * blueNoiseTexturePixel).r;
    float bn_tri = remap_pdf_tri_unity(bn);
    rgb += (bn_tri*2.0-0.5)/255.0;
    #endif

    return rgb;
}

float4 ditherBlueNoise(SAMPLER_TYPE tex, in float4 rgba, float2 fragcoord, const in float time) {
    return float4(ditherBlueNoise(tex, rgba.rgb, fragcoord, time), rgba.a);
}

#if defined(BLUENOISE_TEXTURE)
float ditherBlueNoise(const in float b, float2 fragcoord, const in float time) {
    return ditherBlueNoise(BLUENOISE_TEXTURE, b, fragcoord, time);
}

float3 ditherBlueNoise(const in float3 rgb, float2 fragcoord, const in float time) {
    return ditherBlueNoise(BLUENOISE_TEXTURE, rgb, fragcoord, time);
}

float4 ditherBlueNoise(const in float4 rgba, float2 fragcoord, const in float time) {
    return ditherBlueNoise(BLUENOISE_TEXTURE, rgba, fragcoord, time);
}
#endif

#endif