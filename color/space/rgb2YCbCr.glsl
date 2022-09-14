/*
original_author: Patricio Gonzalez Vivo
description: convert RGB to YCbCr according to https://en.wikipedia.org/wiki/YCbCr
use: rgb2YCbCr(<vec3|vec4> color)
*/

#ifndef FNC_RGB2YCBCR
#define FNC_RGB2YCBCR
vec3 rgb2YCbCr(in vec3 rgb){
    float y = dot(rgb, vec3(.299, .587, .114));
    float cb = .5 + dot(rgb, vec3(-.168736, -.331264, .5));
    float cr = .5 + dot(rgb, vec3(.5, -.418688, -.081312));
    return vec3(y, cb, cr);
}

vec4 rgb2YCbCr(in vec4 rgb) {
    return vec4(rgb2YCbCr(rgb.rgb),rgb.a);
}
#endif