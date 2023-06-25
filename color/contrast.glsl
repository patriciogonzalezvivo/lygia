/*
original_author: Patricio Gonzalez Vivo
description: bias high pass
use: <vec4|vec3|float> contrast(<vec4|vec3|float> value, <float> amount)
*/

#ifndef FNC_CONTRAST
#define FNC_CONTRAST
vec3 contrast(in vec3 value, in float amount) {
    return (value - 0.5 ) * amount + 0.5;
}

vec4 contrast(in vec4 value, in float amount) {
    return vec4(contrast(value.rgb, amount), value.a);
}

float contrast(in float value, in float amount) {
    return (value - 0.5 ) * amount + 0.5;
}
#endif
