#include "../sampler.glsl"
#include "../color/space/rgb2hue.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: samples a hue rainbox pattern color encoded texture and returns a float
use: <fluat> sampleHue(<SAMPLER_TYPE> tex, <vec2> st);
options:
    - SAMPLER_FNC(TEX, UV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SAMPLEHUE_SAMPLE_FNC
#define SAMPLEHUE_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_SAMPLEHUE
#define FNC_SAMPLEHUE
float sampleHue(SAMPLER_TYPE tex, vec2 st) { 
    return rgb2hue( SAMPLER_FNC( tex, st ) ); 
}
#endif