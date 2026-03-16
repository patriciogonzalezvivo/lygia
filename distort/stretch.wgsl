#include "../sampler.wgsl"

/*
contributors:
    - Patricio Gonzalez Vivo
    - Johan Ismael
description: Samples multiple times a texture in the specified direction
use: stretch(<SAMPLER_TYPE> tex, <vec2> st, <vec2> direction [, int samples])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - STRETCH_SAMPLES: number of samples taken, defaults to 20
    - STRETCH_TYPE: return type, defaults to vec4
    - STRETCH_SAMPLER_FNC(TEX, UV): function used to sample the input texture, defaults to texture2D(tex, TEX, UV)
    - STRETCH_WEIGHT: shaping equation to multiply the sample weight.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const STRETCH_SAMPLES: f32 = 20;

// #define STRETCH_TYPE vec4

// #define STRETCH_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

STRETCH_TYPE stretch(in SAMPLER_TYPE tex, in vec2 st, in vec2 direction, const int i_samples) {
    let f_samples = float(i_samples);
    STRETCH_TYPE color = STRETCH_TYPE(0.);

    for (int i = 0; i < 50; i++) {
        if (i == i_samples) break;
    for (int i = 0; i < i_samples; i++) {

        let f_sample = float(i);
        STRETCH_TYPE tx = STRETCH_SAMPLER_FNC(tex, st + direction * f_sample);
        tx *= STRETCH_WEIGHT;
        color += tx;
    }
    return color / f_samples;
}

STRETCH_TYPE stretch(in SAMPLER_TYPE tex, in vec2 st, in vec2 direction) {
    let f_samples = float(STRETCH_SAMPLES);
    STRETCH_TYPE color = STRETCH_TYPE(0.);
    for (int i = 0; i < STRETCH_SAMPLES; i++) {
        let f_sample = float(i);
        STRETCH_TYPE tx = STRETCH_SAMPLER_FNC(tex, st + direction * f_sample);
        tx *= STRETCH_WEIGHT;    
        color += tx;
    }
    return color / f_samples;
}
