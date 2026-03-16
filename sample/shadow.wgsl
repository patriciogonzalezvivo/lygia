#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: sampler shadowMap 
use: 
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <vec4|vec3> _coord)
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <vec2> _coord , float compare)
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <vec2> _shadowMapSize, <vec2> _coord , float compare)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define SAMPLERSHADOW_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r

fn sampleShadow(shadowMap: SAMPLER_TYPE, _coord: vec4f) -> f32 {
    let shadowCoord = _coord.xyz / _coord.w;
    return SAMPLERSHADOW_FNC(shadowMap, shadowCoord.xy);
}

fn sampleShadowa(shadowMap: SAMPLER_TYPE, _coord: vec3f) -> f32 {
    return sampleShadow(shadowMap, vec4f(_coord, 1.0));
}

fn sampleShadowb(shadowMap: SAMPLER_TYPE, uv: vec2f, compare: f32) -> f32 {
    return step(compare, SAMPLERSHADOW_FNC(shadowMap, uv) );
}

fn sampleShadowc(shadowMap: SAMPLER_TYPE, size: vec2f, uv: vec2f, compare: f32) -> f32 {
    return sampleShadow(shadowMap, uv, compare);
}
