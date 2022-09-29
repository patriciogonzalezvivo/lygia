/*
original_author: Patricio Gonzalez Vivo
description: scale a 2D space variable
use: scale(<vec2> st, <vec2|float> scale_factor [, <vec2> center])
options:
    - CENTER
    - CENTER_2D
    - CENTER_3D
*/

#ifndef FNC_SCALE
#define FNC_SCALE
float scale(in float st, in float s, in float center) {
    return (st - center) * s + center;
}

float scale(in float st, in float s) {
    #ifdef CENTER_2D
    return scale(st,  s, CENTER);
    #else
    return scale(st,  s, .5);
    #endif
}


vec2 scale(in vec2 st, in vec2 s, in vec2 center) {
    return (st - center) * s + center;
}

vec2 scale(in vec2 st, in float value, in vec2 center) {
    return scale(st, vec2(value), center);
}

vec2 scale(in vec2 st, in vec2 s) {
    #ifdef CENTER_2D
    return scale(st,  s, CENTER_2D);
    #else
    return scale(st,  s, vec2(.5));
    #endif
}

vec2 scale(in vec2 st, in float value) {
    return scale(st, vec2(value));
}

vec3 scale(in vec3 st, in vec3 s, in vec3 center) {
    return (st - center) * s + center;
}

vec3 scale(in vec3 st, in float value, in vec3 center) {
    return scale(st, vec3(value), center);
}

vec3 scale(in vec3 st, in vec3 s) {
    #ifdef CENTER_3D
    return scale(st,  s, CENTER_3D);
    #else
    return scale(st,  s, vec3(.5));
    #endif
}

vec3 scale(in vec3 st, in float value) {
    return scale(st, vec3(value));
}
#endif
