#include "../space/sprite.hlsl"
#include "../sample.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: sample a frame on a sprite sheet
use: <SAMPLESPRITE_TYPE> sampleSprite(<sampler2D >tex, <float2> st, <float2> grid, <float> frame)
options:
    - SAMPLER_FNC(TEX, UV)
    - SAMPLESPRITE_TYPE: float4
    - SAMPLESPRITE_SAMPLER_FNC(UV)
*/

#ifndef SAMPLESPRITE_TYPE
#define SAMPLESPRITE_TYPE float4
#endif

#ifndef SAMPLESPRITE_SAMPLER_FNC
#define SAMPLESPRITE_SAMPLER_FNC(UV) SAMPLER_FNC(tex, UV)
#endif

#ifndef FNC_SAMPLESPRITE
#define FNC_SAMPLESPRITE

SAMPLESPRITE_TYPE sampleSprite(sampler2D tex, in float2 st, in float2 grid, float frame) {
    return SAMPLESPRITE_SAMPLER_FNC( sprite(st, grid, frame) );
}

#endif