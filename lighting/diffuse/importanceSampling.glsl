#include "../../math/hammersley.glsl"
#include "../../math/rotate3dZ.hlsl"
#include "../../space/tbn.glsl"
#include "../../generative/random.hlsl"
#include "../common/preFilteredImportanceSampling.glsl"

/*
contributors:  Shadi El Hajj
description: Sample the environment map using importance sampling
use: vec3 specularImportanceSampling(float roughness, vec3 f0, const vec3 n, const vec3 v, const vec3 r, const float NoV, const vec2 st)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef IBL_IMPORTANCE_SAMPLING_SAMPLES
#define IBL_IMPORTANCE_SAMPLING_SAMPLES  16
#endif

#if !defined(FNC_DIFFUSE_IMPORTANCE_SAMPLING) &&  __VERSION__ >= 130
#define FNC_DIFFUSE_IMPORTANCE_SAMPLING


vec3 diffuseImportanceSampling(float roughness, vec3 p, vec3 n, vec3 v, vec3 r) {
    const int numSamples = IBL_IMPORTANCE_SAMPLING_SAMPLES;
    const float invNumSamples = 1.0 / float(IBL_IMPORTANCE_SAMPLING_SAMPLES);
    const vec3 up = vec3(0.0, 0.0, 1.0);
    mat3 T = tbn(n, up);
    T *= rotate3dZ(TWO_PI * random(p));

    int width = textureSize(SCENE_CUBEMAP, 0).x;
    float omegaP = (4.0 * PI) / (6.0 * width * width);

    vec3 indirectDiffuse = vec3(0.0, 0.0, 0.0);
    for (int i = 0; i < numSamples; i++) {
        vec2 u = hammersley(i, numSamples);
        vec3 h = T * hemisphereCosSample(u);
        vec3 l = reflect(-v, h);

        float NoL = saturate(dot(n, l));
        if (NoL > 0.0) {
            float ipdf = PI / NoL;
            float mipLevel = prefilteredImportanceSampling(ipdf, omegaP, numSamples) + 1.0;
            vec3 L = textureLod(SCENE_CUBEMAP, l, mipLevel).rgb;
            indirectDiffuse += L;
        }
    }

    return indirectDiffuse * invNumSamples;
}

#endif