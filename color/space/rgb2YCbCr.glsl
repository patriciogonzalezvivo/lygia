/*
contributors: Patricio Gonzalez Vivo
description: Convert RGB to YCbCr according to https://en.wikipedia.org/wiki/YCbCr
use: rgb2YCbCr(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RGB2YCBCR
#define FNC_RGB2YCBCR
vec3 rgb2YCbCr(const in vec3 rgb){
    float y = dot(rgb, vec3(0.299, 0.587, 0.114));
    float cb = .5 + dot(rgb, vec3(-0.168736, -0.331264, 0.5));
    float cr = .5 + dot(rgb, vec3(0.5, -0.418688, -0.081312));
    return vec3(y, cb, cr);
}

vec4 rgb2YCbCr(const in vec4 rgb) {
    return vec4(rgb2YCbCr(rgb.rgb),rgb.a);
}
#endif