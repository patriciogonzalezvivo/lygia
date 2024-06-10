#include "space/hsl2rgb.glsl"
#include "space/rgb2hsl.glsl"

/*
contributors:
    - Johan Ismael
    - Patricio Gonzalez Vivo
description: Shifts color hue
use: hueShift(<vec3|vec4> color, <float> angle)
optionas:
    - HUESHIFT_AMOUNT: if defined, it uses a normalized value instead of an angle
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HUESHIFT
#define FNC_HUESHIFT
vec3 hueShift( vec3 color, float a){
    vec3 hsl = rgb2hsl(color);
#ifndef HUESHIFT_AMOUNT
    hsl.r = hsl.r * TAU + a;
    hsl.r = fract(hsl.r / TAU);
#else
    hsl.r += a;
#endif
    return hsl2rgb(hsl);
}

vec4 hueShift(in vec4 v, in float a) {
    return vec4(hueShift(v.rgb, a), v.a);
}
#endif