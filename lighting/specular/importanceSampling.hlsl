#include "../../math/hammersley.hlsl"
#include "../../space/tbn.hlsl"
#include "../common/ggx.hlsl"
#include "../common/smithGGXCorrelated.hlsl"
#include "../fresnel.hlsl"
#include "../common/preFilteredImportanceSampling.hlsl"

/*
contributors:  Shadi El Hajj
description: Sample the environment map using importance sampling
use: float3 specularImportanceSampling(float roughness, float3 f0, const float3 n, const float3 v, const float3 r, const float NoV, const float2 st)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef IBL_IMPORTANCE_SAMPLING_SAMPLES
#define IBL_IMPORTANCE_SAMPLING_SAMPLES  16
#endif

#ifndef FNC_SPECULAR_IMPORTANCE_SAMPLING
#define FNC_SPECULAR_IMPORTANCE_SAMPLING


float3 specularImportanceSampling(float roughness, float3 f0, const float3 n, const float3 v, const float3 r, const float NoV) {
    const int numSamples = IBL_IMPORTANCE_SAMPLING_SAMPLES;
    const float invNumSamples = 1.0 / float(IBL_IMPORTANCE_SAMPLING_SAMPLES);
    const float3 up = float3(0.0, 0.0, 1.0);
    float3x3 T = tbn(n, up);

    uint width, height, levels;
    SCENE_CUBEMAP.GetDimensions(0, width, height, levels);
    float omegaP = (4.0 * PI) / (6.0 * width * width);

    float3 indirectSpecular = float3(0.0, 0.0, 0.0);
    float dfg2 = 0.0;
    for (int i = 0; i < numSamples; i++) {
        float2 u = hammersley(i, numSamples);
        float3 h = mul(T, importanceSamplingGGX(u, roughness));
        float3 l = lerp(reflect(-v, h), h, roughness);

        float NoL = dot(n, l);
        float NoH = dot(n, h);
        float LoH = max(dot(l, h), EPSILON);

        float D = GGX(n, h, NoH, roughness);
        float V = smithGGXCorrelated_Fast(roughness, NoV, NoL);
        float3 F = fresnel(f0, LoH);

        float ipdf = (4.0 * LoH) / (D * NoH);
        float mipLevel = prefilteredImportanceSampling(ipdf, omegaP, numSamples);
        float3 L = SCENE_CUBEMAP.SampleLevel(SAMPLER_TRILINEAR_CLAMP, l, mipLevel).rgb;

        float3 Fr = F * (D * V * NoL * ipdf * invNumSamples);

        indirectSpecular += (Fr * L);

        dfg2 += F*V*LoH*NoL/NoH;
    }

    dfg2 = 4*dfg2*invNumSamples;
    float3 energyCompensation = 1.0 + f0 * (1.0 / dfg2 - 1.0);
    indirectSpecular *= energyCompensation;

    return indirectSpecular;
}

#endif