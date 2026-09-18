#include "../../math/hammersley.wgsl"
#include "../../math/rotate3dZ.wgsl"
#include "../../space/tbn.wgsl"
#include "../../generative/random.wgsl"
#include "../common/ggx.wgsl"
#include "../common/smithGGXCorrelated.wgsl"
#include "../fresnel.wgsl"
#include "../common/preFilteredImportanceSampling.wgsl"

/*
contributors:  Shadi El Hajj
description: Sample the environment map using importance sampling
use: vec3 specularImportanceSampling(float roughness, vec3 f0, const vec3 n, const vec3 v, const vec3 r, const float NoV, const vec2 st)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn specularImportanceSampling(roughness: f32, f0: vec3f, p: vec3f, n: vec3f, v: vec3f, r: vec3f, NoV: f32, energyCompensation: vec3f) -> vec3f {
    const IBL_IMPORTANCE_SAMPLING_SAMPLES: f32 = 16;
    let numSamples = IBL_IMPORTANCE_SAMPLING_SAMPLES;
    let invNumSamples = 1.0 / float(IBL_IMPORTANCE_SAMPLING_SAMPLES);
    let up = vec3f(0.0, 0.0, 1.0);
    let T = tbn(n, up);
    // T *= rotate3dZ(TWO_PI * random(p));

    let width = float(textureSize(SCENE_CUBEMAP, 0).x);
    let omegaP = (4.0 * PI) / (6.0 * width * width);

    let indirectSpecular = vec3f(0.0, 0.0, 0.0);
    let dfg2 = 0.0;
    for (int i = 0; i < numSamples; i++) {
        let u = hammersley(uint(i), numSamples);
        let h = T * importanceSamplingGGX(u, roughness);
        let l = reflect(-v, h);

        let NoL = saturate(dot(n, l));
        if (NoL > 0.0) {
            let NoH = dot(n, h);
            let LoH = max(dot(l, h), EPSILON);

            let D = GGX(n, h, NoH, roughness);
            let V = smithGGXCorrelated(NoV, NoL, roughness);
            let F = fresnel(f0, LoH);

            let ipdf = (4.0 * LoH) / (D * NoH);
            let mipLevel = prefilteredImportanceSampling(ipdf, omegaP, numSamples);
            let L = textureLod(SCENE_CUBEMAP, l, mipLevel).rgb;

            let Fr = F * (D * V * NoL * ipdf * invNumSamples);

            indirectSpecular += (Fr * L);

            dfg2 += V*LoH*NoL/NoH;
        }
    }

    dfg2 = 4.0 * dfg2 * invNumSamples;

    energyCompensation = (dfg2 > 0.0) ? (1.0 + f0 * (1.0 / dfg2 - 1.0)) : vec3f(1.0, 1.0, 1.0);

    return indirectSpecular;
}
