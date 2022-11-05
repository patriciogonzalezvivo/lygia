/*
original_author: Patricio Gonzalez Vivo
description: change the exposure of a color
use: exposure(<float|vec3|vec4> color, float amount)
*/

#ifndef FNC_EXPOSURE
#define FNC_EXPOSURE
float exposure(float value, float amount) {
    return value * pow(2., amount);
}

vec3 exposure(vec3 color, float amount) {
    return color * pow(2., amount);
}

vec4 exposure(vec4 color, float amount) {
    return vec4(exposure( color.rgb, amount ), color.a);
}
#endif