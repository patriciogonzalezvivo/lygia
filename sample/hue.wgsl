#include "../sampler.wgsl"
#include "../color/space/rgb2hue.wgsl"

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

// #define SAMPLEHUE_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

fn sampleHue(tex: SAMPLER_TYPE, st: vec2f) -> f32 {
    return rgb2hue( SAMPLER_FNC( tex, st ) ); 
}
