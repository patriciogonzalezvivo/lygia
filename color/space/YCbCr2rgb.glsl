/*
contributors: Patricio Gonzalez Vivo
description: Converts YCbCr to RGB according to https://en.wikipedia.org/wiki/YCbCr
use: YCbCr2rgb(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_YCBCR2RGB
#define FNC_YCBCR2RGB
vec3 YCbCr2rgb(const in vec3 ycbcr) {
    float cb = ycbcr.y - .5;
    float cr = ycbcr.z - .5;
    float y = ycbcr.x;
    float r = 1.402 * cr;
    float g = -.344 * cb - .714 * cr;
    float b = 1.772 * cb;
    return vec3(r, g, b) + y;
}

vec4 YCbCr2rgb(const in vec4 ycbcr) {
    return vec4(YCbCr2rgb(ycbcr.rgb),ycbcr.a);
}
#endif
