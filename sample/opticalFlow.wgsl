#include "../sampler.wgsl"
#include "../math/const.wgsl"
#include "../color/space/rgb2hsv.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    sample an optical flow direction where the angle is encoded in the Hue and magnitude in the Saturation
use: sampleFlow(<SAMPLER_TYPE> tex, <vec2> st)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define SAMPLEOPTICALFLOW_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb

fn sampleOpticalFlow(tex: SAMPLER_TYPE, st: vec2f) -> vec2f {
    let data = SAMPLEOPTICALFLOW_SAMPLE_FNC(tex, st);
    let hsv = rgb2hsv(data.rgb);
    let a = (hsv.x * 2.0 - 1.0) * PI;
    return vec2f(cos(a), -sin(a)) * hsv.y;
}

fn sampleOpticalFlowa(tex: SAMPLER_TYPE, st: vec2f, resolution: vec2f, max_dist: f32) -> vec2f { return sampleOpticalFlow(tex, st) * (max_dist / resolution); }
