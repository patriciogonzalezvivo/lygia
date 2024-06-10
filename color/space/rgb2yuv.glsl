/*
contributors: Patricio Gonzalez Vivo
description: Pass a color in RGB and get it in YUB
use: rgb2yuv(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MAT_RGB2YUV
#define MAT_RGB2YUV
#ifdef YUV_SDTV
const mat3 RGB2YUV = mat3(
    0.299, -0.14713,  0.615,
    0.587, -0.28886, -0.51499,
    0.114,  0.436,   -0.10001
);
#else
const mat3 RGB2YUV = mat3(
    0.2126,  -0.09991, 0.615,
    0.7152,  -0.33609,-0.55861,
    0.0722,   0.426,  -0.05639
);
#endif
#endif

#ifndef FNC_RGB2YUV
#define FNC_RGB2YUV
vec3 rgb2yuv(const in vec3 rgb) { return RGB2YUV * rgb; }
vec4 rgb2yuv(const in vec4 rgb) { return vec4(rgb2yuv(rgb.rgb),rgb.a); }
#endif
