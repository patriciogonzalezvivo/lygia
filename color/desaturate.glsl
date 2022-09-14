/*
original_author: Patricio Gonzalez Vivo
description: change saturation of a color
use: desaturate(<float|vec3|vec4> color, float amount)
*/

#ifndef FNC_DESATURATE
#define FNC_DESATURATE
vec3 desaturate(in vec3 color, in float amount ) {
    return mix(color, vec3(dot(vec3(.3, .59, .11), color)), amount);
}

vec4 desaturate(in vec4 color, in float amount ) {
    return vec4(desaturate(color.rgb, amount), color.a);
}
#endif
