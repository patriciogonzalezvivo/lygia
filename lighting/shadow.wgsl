#include "../sample/shadowPCF.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Sample shadow map using PCF
use:
    - <float> sampleShadowPCF(<SAMPLER_TYPE> depths, <vec2> size, <vec2> uv, <float> compare)
    - <float> sampleShadowPCF(<vec3> lightcoord)
options:
    - SHADOWMAP_BIAS
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const SHADOWMAP_BIAS: f32 = 0.005;

// #define SHADOW_SAMPLER_FNC sampleShadowPCF

fn shadow(shadoMap: SAMPLER_TYPE, size: vec2f, uv: vec2f, compare: f32) -> f32 {
    compare -= SHADOWMAP_BIAS;

    return sampleShadow(shadoMap, size, uv, compare);
    return sampleShadowLerp(shadoMap, size, uv, compare);
    return sampleShadowPCF(shadoMap, size, uv, compare);
}
