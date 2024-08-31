#include "../../math/hammersley.glsl"
#include "../../space/tbn.glsl"
#include "../common/ggx.glsl"
#include "../common/smithGGXCorrelated.glsl"
#include "../fresnel.glsl"
#include "../common/preFilteredImportanceSampling.glsl"

/*
contributors:  Shadi El Hajj
description: Sample the environment map using importance sampling
use: vec3 specularImportanceSampling(float roughness, vec3 f0, const vec3 n, const vec3 v, const vec3 r, const float NoV, const float2 st)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef IBL_IMPORTANCE_SAMPLING_SAMPLES
#define IBL_IMPORTANCE_SAMPLING_SAMPLES  64
#endif

#ifndef FNC_SPECULAR_IMPORTANCE_SAMPLING
#define FNC_SPECULAR_IMPORTANCE_SAMPLING


vec3 specularImportanceSampling(float roughness, vec3 f0, const vec3 n, const vec3 v, const vec3 r, const float NoV) {
    const int numSamples = IBL_IMPORTANCE_SAMPLING_SAMPLES;
    const float invNumSamples = 1.0 / float(IBL_IMPORTANCE_SAMPLING_SAMPLES);
    const vec3 up = vec3(0.0, 0.0, 1.0);
    vec3x3 T = tbn(n, up);

    uint width = textureSize(SCENE_CUBEMAP, 0);
    float omegaP = (4.0 * PI) / (6.0 * width * width);

    vec3 indirectSpecular = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < numSamples; i++) {
        float2 u = hammersley(i, numSamples);
        vec3 h = mul(T, importanceSamplingGGX(u, roughness));
        vec3 l = lerp(r, n, roughness * roughness);

        float NoL = saturate(dot(n, l));
        if (NoL > 0.0) {
            float NoH = dot(n, h);
            float LoH = saturate(dot(l, h));

            float D = GGX(n, h, NoH, roughness);
            float V = smithGGXCorrelated_Fast(roughness, NoV, NoL);
            vec3 F = fresnel(f0, LoH);

            float ipdf = (4.0 * LoH) / (D * NoH);
            float mipLevel = prefilteredImportanceSampling(ipdf, omegaP, numSamples);
            vec3 L = SCENE_CUBEMAP.SampleLevel(SAMPLER_TRILINEAR_CLAMP, l, mipLevel).rgb;

            vec3 Fr = F * (D * V * NoL * ipdf * invNumSamples);

            indirectSpecular += (Fr * L);
        }
    }

    return indirectSpecular;
}

#endif