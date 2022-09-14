#include "space/rgb2luma.glsl"

/*
original_author: Hugh Kennedy (https://github.com/hughsk)
description: get the luminosity of a color. From https://github.com/hughsk/glsl-luma/blob/master/index.glsl
use: luma(<vec3|vec4> color)
*/

#ifndef FNC_LUMA
#define FNC_LUMA
float luma(float color) {
    return color;
}

float luma(in vec3 color) {
    return rgb2luma(color);
}

float luma(in vec4 color) {
    return rgb2luma(color.rgb);
}
#endif
