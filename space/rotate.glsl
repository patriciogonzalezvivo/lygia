#include "../math/rotate2d.glsl"
#include "../math/rotate4d.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian radians
use: rotate(<vec3|vec2> st, float radians [, vec2 center])
options:
    - CENTER_2D
    - CENTER_3D
    - CENTER_4D
*/

#ifndef FNC_ROTATE
#define FNC_ROTATE
vec2 rotate(in vec2 st, in float radians, in vec2 center) {
    return rotate2d(radians) * (st - center) + center;
}

vec2 rotate(in vec2 st, in float radians) {
    #ifdef CENTER_2D
    return rotate(st, radians, CENTER_2D);
    #else
    return rotate(st, radians, vec2(.5));
    #endif
}

vec2 rotate(vec2 st, vec2 x_axis) {
    #ifdef CENTER_2D
    st -= CENTER_2D;
    #endif
    vec2 rta = vec2( dot(st, vec2(-x_axis.y, x_axis.x)), dot(st, x_axis) );
    #ifdef CENTER_2D
    rta += CENTER_2D;
    #endif
    return rta;
}

vec3 rotate(in vec3 xyz, in float radians, in vec3 axis, in vec3 center) {
    return (rotate4d(axis, radians) * vec4(xyz - center, 1.)).xyz + center;
}

vec3 rotate(in vec3 xyz, in float radians, in vec3 axis) {
    #ifdef CENTER_3D
    return rotate(xyz, radians, axis, CENTER_3D);
    #else
    return rotate(xyz, radians, axis, vec3(0.));
    #endif
}

vec4 rotate(in vec4 xyzw, in float radians, in vec3 axis, in vec4 center) {
    return rotate4d(axis, radians) * (xyzw - center) + center;
}

vec4 rotate(in vec4 xyzw, in float radians, in vec3 axis) {
    #ifdef CENTER_4D
    return rotate(xyzw, radians, axis, CENTER_4D);
    #else
    return rotate(xyzw, radians, axis, vec4(0.));
    #endif
}
#endif
