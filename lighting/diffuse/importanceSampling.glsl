#include "../../math/hammersley.glsl"
#include "../../space/tbn.glsl"
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

#ifndef FNC_DIFFUSE_IMPORTANCE_SAMPLING
#define FNC_DIFFUSE_IMPORTANCE_SAMPLING


vec3 diffuseImportanceSampling(float roughness, const vec3 n, const vec3 v, const vec3 r) {
    const int numSamples = IBL_IMPORTANCE_SAMPLING_SAMPLES;
    const float invNumSamples = 1.0 / float(IBL_IMPORTANCE_SAMPLING_SAMPLES);
    const vec3 up = vec3(0.0, 0.0, 1.0);
    vec3x3 T = tbn(n, up);

    uint width = textureSize(SCENE_CUBEMAP, 0);
    float omegaP = (4.0 * PI) / (6.0 * width * width);

    vec3 indirectDiffuse = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < numSamples; i++) {
        float2 u = hammersley(i, numSamples);
        vec3 h = mul(T, hemisphereCosSample(u));

        vec3 l = n;
        float NoL = saturate(dot(n, l));
        if (NoL > 0.0) {
            float ipdf = PI / NoL;
            float mipLevel = prefilteredImportanceSampling(ipdf, omegaP, numSamples) + 1.0;
            vec3 L = SCENE_CUBEMAP.SampleLevel(SAMPLER_TRILINEAR_CLAMP, l, mipLevel).rgb;
            indirectDiffuse += L;
        }
    }

    return indirectDiffuse * invNumSamples;
}

#endif