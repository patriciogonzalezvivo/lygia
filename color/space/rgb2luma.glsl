/*
original_author: Hugh Kennedy (https://github.com/hughsk)
description: get's the luminosity of a color. From https://github.com/hughsk/glsl-luma/blob/master/index.glsl
use: rgb2luma(<vec3|vec4> color)
*/

#ifndef FNC_RGB2LUMA
#define FNC_RGB2LUMA
float rgb2luma(in vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

float rgb2luma(in vec4 color) {
    return rgb2luma(color.rgb);
}
#endif
