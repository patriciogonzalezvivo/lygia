/*
original_author: Patricio Gonzalez Vivo
description: pass a color in RGB and get it in YUB
use: rgb2yuv(<vec3|vec4> color)
*/

#ifndef FNC_RGB2YUV
#define FNC_RGB2YUV

#ifdef YUV_SDTV
const mat3 rgb2yuv_mat = mat3(
    .299, -.14713,  .615,
    .587, -.28886, -.51499,
    .114,  .436,   -.10001
);
#else
const mat3 rgb2yuv_mat = mat3(
    .2126,  -.09991, .615,
    .7152,  -.33609,-.55861,
    .0722,   .426,  -.05639
);
#endif

vec3 rgb2yuv(in vec3 rgb) {
    return rgb2yuv_mat * rgb;
}

vec4 rgb2yuv(in vec4 rgb) {
    return vec4(rgb2yuv(rgb.rgb),rgb.a);
}
#endif
