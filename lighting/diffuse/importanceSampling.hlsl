#include "../../math/hammersley.hlsl"
#include "../../math/rotate3dZ.hlsl"
#include "../../space/tbn.hlsl"
#include "../../generative/random.hlsl"
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

#ifndef FNC_DIFFUSE_IMPORTANCE_SAMPLING
#define FNC_DIFFUSE_IMPORTANCE_SAMPLING


float3 diffuseImportanceSampling(float roughness, float3 p, float3 n, float3 v, float3 r) {
    const int numSamples = IBL_IMPORTANCE_SAMPLING_SAMPLES;
    const float invNumSamples = 1.0 / float(IBL_IMPORTANCE_SAMPLING_SAMPLES);
    const float3 up = float3(0.0, 0.0, 1.0);
    float3x3 T = tbn(n, up);
    T = mul(T, rotate3dZ(TWO_PI * random(p)));

    uint width, height, levels;
    SCENE_CUBEMAP.GetDimensions(0, width, height, levels);
    float omegaP = (4.0 * PI) / (6.0 * width * width);

    float3 indirectDiffuse = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < numSamples; i++) {
        float2 u = hammersley(i, numSamples);
        float3 h = mul(T, hemisphereCosSample(u));
        float3 l = lerp(reflect(-v, h), h, roughness);

        float NoL = saturate(dot(n, l));
        if (NoL > 0.0) {
            float ipdf = PI / NoL;
            float mipLevel = prefilteredImportanceSampling(ipdf, omegaP, numSamples) + 1.0;
            float3 L = SCENE_CUBEMAP.SampleLevel(SAMPLER_TRILINEAR_CLAMP, l, mipLevel).rgb;
            indirectDiffuse += L;
        }
    }

    return indirectDiffuse * invNumSamples;
}

#endif