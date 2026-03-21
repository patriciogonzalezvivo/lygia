#include "../color/space/YCbCr2rgb.wgsl"
#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: a way to solve the YUV camare texture on iOS
use: sampleYUV(<SAMPLER_TYPE> tex1, <SAMPLER_TYPE> tex2, <vec2> st)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn sampleYUV(tex_Y: SAMPLER_TYPE, tex_UV: SAMPLER_TYPE, st: vec2f) -> vec4f {
    let yuv = vec4f(0.);
    yuv.xw = SAMPLER_FNC(tex_Y, st).ra;
    yuv.yz = SAMPLER_FNC(tex_UV, st).rg;
    return YCbCr2rgb(yuv);
}
