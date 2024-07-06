#include "space/srgb2rgb.hlsl"
#include "space/rgb2srgb.hlsl"

#include "space/oklab2rgb.hlsl"
#include "space/rgb2oklab.hlsl"

/*
contributors: [Bjorn Ottosson, Inigo Quiles]
description: |
    Mix function by Inigo Quiles (https://www.shadertoy.com/view/ttcyRS) 
    utilizing Bjorn Ottosso's OkLab color space, which is provide smooth stransitions 
    Learn more about it [his article](https://bottosson.github.io/posts/oklab/)
use: <float3\float4> mixOklab(<float3|float4> colorA, <float3|float4> colorB, float pct)
options:
    - MIXOKLAB_SRGB: by default colA and colB use linear RGB. If you want to use sRGB define this flag
examples:
    - /shaders/color_mix.frag
license: 
    - MIT License (MIT) Copyright (c) 2020 Björn Ottosson
    - MIT License (MIT) Copyright (c) 2020 Inigo Quilez
*/

#ifndef FNC_MIXOKLAB
#define FNC_MIXOKLAB
float3 mixOklab( float3 colA, float3 colB, float h ) {

    #ifdef MIXOKLAB_SRGB
    colA = srgb2rgb(colA);
    colB = srgb2rgb(colB);
    #endif

    float3 lmsA = pow(mul(RGB2OKLAB_B, colA), float3(0.33333, 0.33333, 0.33333));
    float3 lmsB = pow(mul(RGB2OKLAB_B, colB), float3(0.33333, 0.33333, 0.33333));
    float3 lms = lerp( lmsA, lmsB, h );
    
    // cone to rgb
    float3 rgb = mul(OKLAB2RGB_B, lms*lms*lms);

    #ifdef MIXOKLAB_SRGB
    return rgb2srgb(rgb);
    #else
    return rgb;
    #endif
}

float4 mixOklab( float4 colA, float4 colB, float h ) {
    return float4( mixOklab(colA.rgb, colB.rgb, h), lerp(colA.a, colB.a, h) );
}
#endif