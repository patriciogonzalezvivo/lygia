/*
original_author: Patricio Gonzalez Vivo
description: |
    Convert from linear RGB to YIQ which was the followin range. 
    Using conversion matrices from FCC NTSC Standard (SMPTE C) https://en.wikipedia.org/wiki/YIQ
use: rgb2yiq(<vec3|vec4> color)
*/

#ifndef FNC_RGB2YIQ
#define FNC_RGB2YIQ

const mat3 rgb2yiq_mat = mat3(0.300,  0.5900,  0.1100, 
                              0.599, -0.2773, -0.3217, 
                              0.213, -0.5251,  0.3121);

vec3 rgb2yiq(in vec3 rgb) {
  return rgb2yiq_mat * rgb;
}

vec4 rgb2yiq(in vec4 rgb) {
    return vec4(rgb2yiq(rgb.rgb), rgb.a);
}
#endif
