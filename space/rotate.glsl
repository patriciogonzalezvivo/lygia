#include "../math/rotate2d.glsl"
#include "../math/rotate4d.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian r
use: rotate(<vec3|vec2> v, float r [, vec2 c])
options:
    - CENTER_2D
    - CENTER_3D
    - CENTER_4D
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATE
#define FNC_ROTATE
vec2 rotate(in vec2 v, in float r, in vec2 c) {
    return rotate2d(r) * (v - c) + c;
}

vec2 rotate(in vec2 v, in float r) {
    #ifdef CENTER_2D
    return rotate(v, r, CENTER_2D);
    #else
    return rotate(v, r, vec2(.5));
    #endif
}

vec2 rotate(vec2 v, vec2 x_axis) {
    #ifdef CENTER_2D
    v -= CENTER_2D;
    #endif
    vec2 rta = vec2( dot(v, vec2(-x_axis.y, x_axis.x)), dot(v, x_axis) );
    #ifdef CENTER_2D
    rta += CENTER_2D;
    #endif
    return rta;
}

vec3 rotate(in vec3 v, in float r, in vec3 axis, in vec3 c) {
    return (rotate4d(axis, r) * vec4(v - c, 1.)).xyz + c;
}

vec3 rotate(in vec3 v, in float r, in vec3 axis) {
    #ifdef CENTER_3D
    return rotate(v, r, axis, CENTER_3D);
    #else
    return rotate(v, r, axis, vec3(0.));
    #endif
}

vec4 rotate(in vec4 v, in float r, in vec3 axis, in vec4 c) {
    return rotate4d(axis, r) * (v - c) + c;
}

vec4 rotate(in vec4 v, in float r, in vec3 axis) {
    #ifdef CENTER_4D
    return rotate(v, r, axis, CENTER_4D);
    #else
    return rotate(v, r, axis, vec4(0.));
    #endif
}

#if defined(FNC_QUATMULT)
vec3 rotate(QUAT q, vec3 v) {
    QUAT q_c = QUAT(-q.x, -q.y, -q.z, q.w);
    return quatMul(q, quatMul(vec4(v, 0), q_c)).xyz;
}

vec3 rotate(QUAT q, vec3 v, vec3 c) {
    vec3 dir = v - c;
    return c + rotate(q, dir);
}
#endif

#endif
