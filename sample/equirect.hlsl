#include "../space/xyz2equirect.hlsl"
#include "../generative/random.hlsl"
#include "../generative/srandom.hlsl"
#include "../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample an equirect texture as it was a cubemap
use: sampleEquirect(<SAMPLER_TYPE> texture, <float3> dir)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLEEQUIRET_ITERATIONS: null
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLEEQUIRECT
#define FNC_SAMPLEEQUIRECT
float4 sampleEquirect(SAMPLER_TYPE tex, float3 dir) { 
    float2 st = xyz2equirect(dir);
    #ifdef SAMPLEEQUIRECT_FLIP_Y
    st.y = 1.0-st.y;
    #endif
    return SAMPLER_FNC(tex, st); 
}

float4 sampleEquirect(SAMPLER_TYPE tex, float3 dir, float lod) { 
    
    #if defined(SAMPLEEQUIRET_ITERATIONS)
    float4 acc = float4(0.0, 0.0, 0.0, 0.0);
    float2 st = xyz2equirect(dir);
    #ifdef SAMPLEEQUIRECT_FLIP_Y
    st.y = 1.0-st.y;
    #endif
    float2x2 rot = float2x2(cos(GOLDEN_ANGLE), sin(GOLDEN_ANGLE), -sin(GOLDEN_ANGLE), cos(GOLDEN_ANGLE));
    float r = 1.;
    float2 vangle = float2(0.0, lod * 0.01);
    for (int i = 0; i < SAMPLEEQUIRET_ITERATIONS; i++) {
        vangle = mul(rot, vangle);
        r++;
        float4 col = SAMPLER_FNC(tex, st + random( float3(st, r) ) * vangle );
        acc += col * col;
    }
    return float4(acc.rgb/acc.a, 1.0); 

    #else
    dir += srandom3( dir ) * 0.01 * lod;
    float2 st = xyz2equirect(dir);
    #ifdef SAMPLEEQUIRECT_FLIP_Y
    st.y = 1.0-st.y;
    #endif
    return SAMPLER_FNC(tex, st);

    #endif
}

#endif