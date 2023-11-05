#include "space/rgb2hsv.glsl"
#include "space/hsv2rgb.glsl"

/*
contributors: Johan Ismael
description: shifts color hue
use: hueShift(<vec3|vec4> color, <float> amount)
*/

#ifndef FNC_HUESHIFT
#define FNC_HUESHIFT
vec3 hueShift(in vec3 v, in float a) {
    vec3 hsv = rgb2hsv(v);
    hsv.r += a;
    return hsv2rgb(hsv);
}

vec4 hueShift(in vec4 v, in float a) {
    return vec4(hueShift(v.rgb, a), v.a);
}
#endif