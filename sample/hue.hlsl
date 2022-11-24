#include "../sample.hlsl"
#include "../color/space/rgb2hue.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: samples a hue rainbox pattern color encoded texture and returns a float
use: <fluat> sampleHue(<sampler2D> tex, <float2> st);
options:
    - SAMPLER_FNC(TEX, UV)
*/

#ifndef SAMPLEHUE_SAMPLE_FNC
#define SAMPLEHUE_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_SAMPLEHUE
#define FNC_SAMPLEHUE
float sampleHue(sampler2D tex, float2 st) { 
    return rgb2hue( SAMPLER_FNC( tex, st ) ); 
}
#endif