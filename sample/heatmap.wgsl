#include "../sampler.wgsl"
#include "../color/space/rgb2heat.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample a value encoded on a heatmap
use: sampleFlow(<SAMPLER_TYPE> tex, <vec2> st)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define SAMPLEHEATMAP_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb

fn sampleHeatmap(tex: SAMPLER_TYPE, st: vec2f) -> f32 { return rgb2heat( SAMPLEHEATMAP_SAMPLE_FNC(tex, st) ); }
fn sampleHeatmapa(tex: SAMPLER_TYPE, st: vec2f, _min: f32, _max: f32) -> f32 { return  _min + sampleHeatmap(tex, st) * (_max - _min); }
