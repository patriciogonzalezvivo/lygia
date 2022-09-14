/*
original_author: Patricio Gonzalez Vivo
description: pass a color in YUB and get RGB color
use: yuv2rgb(<vec3|vec4> color)
*/

#ifndef FNC_YUV2RGB
#define FNC_YUV2RGB

#ifdef YUV_SDTV
const mat3 yuv2rgb_mat = mat3(
    1.,       1. ,      1.,
    0.,       -.39465,  2.03211,
    1.13983,  -.58060,  0.
);
#else
const mat3 yuv2rgb_mat = mat3(
    1.,       1. ,      1.,
    0.,       -.21482,  2.12798,
    1.28033,  -.38059,  0.
);
#endif

vec3 yuv2rgb(in vec3 yuv) {
    return yuv2rgb_mat * yuv;
}

vec4 yuv2rgb(in vec4 yuv) {
    return vec4(yuv2rgb(yuv.rgb), yuv.a);
}
#endif
