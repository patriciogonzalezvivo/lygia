#include "../color/space/YCbCr2rgb.glsl"
#include "../sample.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: a way to solve the YUV camare texture on iOS
use: sampleYUV(<sampler2D> tex1, <sampler2D> tex2, <vec2> st)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
*/

#ifndef FNC_SAMPLEYUV
#define FNC_SAMPLEYUV
vec4 sampleYUV(in sampler2D tex_Y, in sampler2D tex_UV, in vec2 st) {
    vec4 yuv = vec4(0.);
    yuv.xw = SAMPLER_FNC(tex_Y, st).ra;
    yuv.yz = SAMPLER_FNC(tex_UV, st).rg;
    return YCbCr2rgb(yuv);
}
#endif
