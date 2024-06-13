#include "hueShift.glsl"
#include "space/rgb2ryb.glsl"
#include "space/ryb2rgb.glsl"

/*
contributors:
    - Johan Ismael
    - Patricio Gonzalez Vivo
description: Shifts color hue in the RYB color space
use: hueShift(<vec3|vec4> color, <float> angle)
optionas:
    - HUESHIFT_AMOUNT: if defined, it uses a normalized value instead of an angle
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HUESHIFTRYB
#define FNC_HUESHIFTRYB
vec3 hueShiftRYB( vec3 color, float a){
    vec3 rgb = rgb2ryb(color);
    rgb = hueShift(rgb, PI);
    return ryb2rgb(rgb);
}

vec4 hueShiftRYB(in vec4 v, in float a) {
    return vec4(hueShiftRYB(v.rgb, a), v.a);
}
#endif