/*
contributors: Patricio Gonzalez Vivo
description: Pass a color in YUB and get RGB color
use: yuv2rgb(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MAT_YUV2RGB
#define MAT_YUV2RGB
#ifdef YUV_SDTV
const mat3 YUV2RGB = mat3(
    1.0,       1.0,      1.0,
    0.0,      -0.39465,  2.03211,
    1.13983,  -0.58060,  0.0
);
#else
const mat3 YUV2RGB = mat3(
    1.0,       1.0,      1.0,
    0.0,      -0.21482,  2.12798,
    1.28033,  -0.38059,  0.0
);
#endif
#endif

#ifndef FNC_YUV2RGB
#define FNC_YUV2RGB
vec3 yuv2rgb(const in vec3 yuv) { return YUV2RGB * yuv; }
vec4 yuv2rgb(const in vec4 yuv) { return vec4(yuv2rgb(yuv.rgb), yuv.a); }
#endif
