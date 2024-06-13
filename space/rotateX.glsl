#include "../math/rotate4dX.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateX(<vec3|vec4> v, float radian [, vec3 center])
options:
    - CENTER_3D
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATEX
#define FNC_ROTATEX
vec4 rotateX(in vec4 v, in float r, in vec4 c) {
    return rotate4dX(r) * (v - c) + c;
}

vec4 rotateX(in vec4 v, in float r) {
    #ifdef CENTER_4D
    return rotate4dX(r) * (v - CENTER_4D) + CENTER_4D;
    #else
    return rotate4dX(r) * v;
    #endif
}

vec3 rotateX(in vec3 v, in float r, in vec3 c) {
    return (rotate4dX(r) * vec4(v - c, 1.0)).xyz + c;
}

vec3 rotateX(in vec3 v, in float r) {
    #ifdef CENTER_3D
    return (rotate4dX(r) * vec4(v - CENTER_3D, 1.0)).xyz + CENTER_3D;
    #else
    return (rotate4dX(r) * vec4(v, 1.0)).xyz;
    #endif
}
#endif
