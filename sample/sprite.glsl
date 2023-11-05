#include "../space/sprite.glsl"
#include "../sample.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample a frame on a sprite sheet
use: <SAMPLESPRITE_TYPE> sampleSprite(<SAMPLER_TYPE >tex, <vec2> st, <vec2> grid, <float> frame)
options:
    - SAMPLER_FNC(TEX, UV)
    - SAMPLESPRITE_TYPE: vec4
    - SAMPLESPRITE_SAMPLER_FNC(UV)
*/

#ifndef SAMPLESPRITE_TYPE
#define SAMPLESPRITE_TYPE vec4
#endif

#ifndef SAMPLESPRITE_SAMPLER_FNC
#define SAMPLESPRITE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_SAMPLESPRITE
#define FNC_SAMPLESPRITE

SAMPLESPRITE_TYPE sampleSprite(SAMPLER_TYPE tex, in vec2 st, in vec2 grid, float frame) {
    return SAMPLESPRITE_SAMPLER_FNC(tex, sprite(st, grid, frame) );
}

#endif