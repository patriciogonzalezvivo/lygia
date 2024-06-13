#include "../math/rotate4dZ.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateZ(<vec3|vec4> pos, float radian [, vec3 center])
options:
    - CENTER_3D
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATEZ
#define FNC_ROTATEZ
vec4 rotateZ(in vec4 v, in float r, in vec4 c) {
    return rotate4dZ(r) * (v - c) + c;
}

vec4 rotateZ(in vec4 v, in float r) {
    #ifdef CENTER_4D
    return rotate4dZ(r) * (v - CENTER_3D) + CENTER_3D;
    #else
    return rotate4dZ(r) * v;
    #endif
}

vec3 rotateZ(in vec3 v, in float r, in vec3 c) {
    return (rotate4dZ(r) * vec4(v - c, 0.0) ).xyz + c;
}

vec3 rotateZ(in vec3 v, in float r) {
    #ifdef CENTER_3D
    return (rotate4dZ(r) * vec4(v - CENTER_3D, 0.0)).xyz + CENTER_3D;
    #else
    return (rotate4dZ(r) * vec4(v, 0.0)).xyz;
    #endif
}
#endif
