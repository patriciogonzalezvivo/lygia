#include "../../math/hammersley.glsl"
#include "../../math/rotate3dZ.glsl"
#include "../../space/tbn.glsl"
#include "../../generative/random.glsl"
#include "../common/ggx.glsl"
#include "../common/smithGGXCorrelated.glsl"
#include "../fresnel.glsl"
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

#if !defined(FNC_SPECULAR_IMPORTANCE_SAMPLING) && defined(SCENE_CUBEMAP) &&  __VERSION__ >= 130 
#define FNC_SPECULAR_IMPORTANCE_SAMPLING

vec3 specularImportanceSampling(float roughness, vec3 f0, vec3 p, vec3 n, vec3 v, vec3 r, float NoV, out vec3 energyCompensation) {
    const int numSamples = IBL_IMPORTANCE_SAMPLING_SAMPLES;
    const float invNumSamples = 1.0 / float(IBL_IMPORTANCE_SAMPLING_SAMPLES);
    const vec3 up = vec3(0.0, 0.0, 1.0);
    mat3 T = tbn(n, up);
    // T *= rotate3dZ(TWO_PI * random(p));

    float width = float(textureSize(SCENE_CUBEMAP, 0).x);
    float omegaP = (4.0 * PI) / (6.0 * width * width);

    vec3 indirectSpecular = vec3(0.0, 0.0, 0.0);
    float dfg2 = 0.0;
    for (int i = 0; i < numSamples; i++) {
        vec2 u = hammersley(uint(i), numSamples);
        vec3 h = T * importanceSamplingGGX(u, roughness);
        vec3 l = reflect(-v, h);

        float NoL = saturate(dot(n, l));
        if (NoL > 0.0) {
            float NoH = dot(n, h);
            float LoH = max(dot(l, h), EPSILON);

            float D = GGX(n, h, NoH, roughness);
            float V = smithGGXCorrelated(NoV, NoL, roughness);
            vec3 F = fresnel(f0, LoH);

            float ipdf = (4.0 * LoH) / (D * NoH);
            float mipLevel = prefilteredImportanceSampling(ipdf, omegaP, numSamples);
            vec3 L = textureLod(SCENE_CUBEMAP, l, mipLevel).rgb;

            vec3 Fr = F * (D * V * NoL * ipdf * invNumSamples);

            indirectSpecular += (Fr * L);

            dfg2 += V*LoH*NoL/NoH;
        }
    }

    dfg2 = 4.0 * dfg2 * invNumSamples;

    energyCompensation = (dfg2 > 0.0) ? (1.0 + f0 * (1.0 / dfg2 - 1.0)) : vec3(1.0, 1.0, 1.0);

    return indirectSpecular;
}

#endif