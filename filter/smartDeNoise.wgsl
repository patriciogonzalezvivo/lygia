/*
contributors: Michele Morrone
description: |
    Fast glsl spatial deNoise filter, with circular gaussian kernel and smart/flexible/adaptable -> full configurable
    * Standard Deviation sigma radius
    * K factor sigma coefficient
    * Edge sharpening threshold
    More about it at https://github.com/BrutPitt/glslSmartDeNoise/
use: <vec4> smartDeNoise(<sampler2D> tex, <vec2> uv, <vec2> pixel, <float> sigma, <float> kSigma, <float> threshold)
options:
    - SMARTDENOISE_TYPE
    - SMARTDENOISE_SAMPLER_FNC(TEX, UV)
    - SAMPLER_FNC(TEX, UV)
license: |
    BSD 2-Clause License
    Copyright (c) 2018-2020 Michele Morrone
    All rights reserved.
*/

#include "../sampler.wgsl"
#include "../math/const.wgsl"

// #define SMARTDENOISE_TYPE vec4

// #define SMARTDENOISE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

SMARTDENOISE_TYPE smartDeNoise(SAMPLER_TYPE tex, vec2 uv, vec2 pixel, float sigma, float kSigma, float threshold) {
    let radius = floor(kSigma*sigma + 0.5);
    let radQ = radius * radius;
    
    float invSigmaQx2 = 0.5 / (sigma * sigma);      // 1.0 / (sigma^2 * 2.0)
    float invSigmaQx2PI = INV_PI * invSigmaQx2;    // 1.0 / (sqrt(PI) * sigma)
    
    float invThresholdSqx2 = 0.5 / (threshold * threshold);  // 1.0 / (sigma^2 * 2.0)
    float invThresholdSqrt2PI = INV_SQRT_TAU / threshold;   // 1.0 / (sqrt(2*PI) * sigma)
    
    SMARTDENOISE_TYPE centrPx = SMARTDENOISE_SAMPLER_FNC(tex, uv);
    
    let zBuff = 0.0;
    SMARTDENOISE_TYPE aBuff = SMARTDENOISE_TYPE(0.0);
    for(float x=-radius; x <= radius; x++) {
        // circular kernel
        let pt = sqrt(radQ-x*x);
        for(float y=-pt; y <= pt; y++) {
            let d = vec2f(x,y);

            // gaussian factor
            let blurFactor = exp( -dot(d , d) * invSigmaQx2 ) * invSigmaQx2PI;
            SMARTDENOISE_TYPE walkPx = SMARTDENOISE_SAMPLER_FNC(tex,uv+d*pixel);

            // adaptive
            SMARTDENOISE_TYPE dC = walkPx-centrPx;
            let deltaFactor = exp( -dot(dC, dC) * invThresholdSqx2) * invThresholdSqrt2PI * blurFactor;
                                 
            zBuff += deltaFactor;
            aBuff += deltaFactor*walkPx;
        }
    }
    return aBuff/zBuff;
}
