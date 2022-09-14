#include "space/rgb2hsv.glsl"
#include "space/hsv2rgb.glsl"

/*
original_author: Johan Ismael
description: shifts color hue
use: hueShift(<vec3|vec4> color, <float> amount)
*/

#ifndef FNC_HUESHIFT
#define FNC_HUESHIFT
vec3 hueShift(in vec3 color, in float amount) {
    vec3 hsv = rgb2hsv(color);
    hsv.r += amount;
    return hsv2rgb(hsv);
}

vec4 hueShift(in vec4 color, in float amount) {
    return vec4(hueShift(color.rgb, amount), color.a);
}
#endif