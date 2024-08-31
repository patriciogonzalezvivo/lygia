#include "../../math/hammersley.hlsl"
#include "../../space/tbn.hlsl"
#include "../common/ggx.hlsl"
#include "../common/smithGGXCorrelated.hlsl"
#include "../fresnel.hlsl"

/*
contributors:  Shadi El Hajj
description: Sample the environment map using importance sampling
use: float3 specularImportanceSampling(float roughness, float3 f0, const float3 n, const float3 v, const float3 r, const float NoV, const float2 st)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef IBL_IMPORTANCE_SAMPLING_SAMPLES
#define IBL_IMPORTANCE_SAMPLING_SAMPLES  64
#endif

#ifndef FNC_SPECULAR_IMPORTANCE_SAMPLING
#define FNC_SPECULAR_IMPORTANCE_SAMPLING

float prefilteredImportanceSampling(float ipdf, float omegaP) {
    const float numSamples = float(IBL_IMPORTANCE_SAMPLING_SAMPLES);
    const float invNumSamples = 1.0 / float(numSamples);
    const float K = 4.0;
    float omegaS = invNumSamples * ipdf;
    float mipLevel = log2(K * omegaS / omegaP) * 0.5; // log4
    return mipLevel;
}

float3 specularImportanceSampling(float roughness, float3 f0, const float3 n, const float3 v, const float3 r, const float NoV) {
    const int numSamples = IBL_IMPORTANCE_SAMPLING_SAMPLES;
    const float invNumSamples = 1.0 / float(IBL_IMPORTANCE_SAMPLING_SAMPLES);
    const float3 up = float3(0.0, 0.0, 1.0);

    // tangent space
    float3 T0 = normalize(cross(up, n));
    float3x3 T = tbn(n, up);

    uint width, height, levels;
    SCENE_CUBEMAP.GetDimensions(0, width, height, levels);
    float omegaP = (4.0 * PI) / (6.0 * width * width);

    float3 indirectSpecular = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < numSamples; i++) {
        float2 u = hammersley(i, IBL_IMPORTANCE_SAMPLING_SAMPLES);
        float3 h = mul(T, importanceSamplingGGX(u, roughness));
        float3 l = lerp(r, n, roughness * roughness);

        float NoL = saturate(dot(n, l));
        if (NoL > 0.0) {
            float NoH = dot(n, h);
            float LoH = saturate(dot(l, h));

            float D = GGX(n, h, NoH, roughness);
            float V = smithGGXCorrelated_Fast(NoV, NoL, roughness);
            float3 F = fresnel(f0, LoH);

            float ipdf = (4.0 * LoH) / (D * NoH);
            float mipLevel = prefilteredImportanceSampling(ipdf, omegaP);
            float3 L = SCENE_CUBEMAP.SampleLevel(SAMPLER_TRILINEAR_CLAMP, l, mipLevel).rgb;

            float3 Fr = F * (D * V * NoL * ipdf * invNumSamples);

            indirectSpecular += (Fr * L);
        }
    }

    return indirectSpecular;
}

#endif