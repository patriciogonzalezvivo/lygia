#include "../math/rotate4dY.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateY(<vec3> pos, float radian [, vec4 center])
options:
    - CENTER_3D
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ROTATEY
#define FNC_ROTATEY
vec4 rotateY(in vec4 v, in float r, in vec4 c) {
    return rotate4dY(r) * (v - c) + c;
}

vec4 rotateY(in vec4 v, in float r) {
    #ifdef CENTER_4D
    return rotate4dY(r) * (v - CENTER_4D) + CENTER_4D;
    #else
    return rotate4dY(r) * v;
    #endif
}

vec3 rotateY(in vec3 v, in float r, in vec3 c) {
    return (rotate4dY(r) * vec4(v - c, 1.)).xyz + c;
}

vec3 rotateY(in vec3 v, in float r) {
    #ifdef CENTER_3D
    return (rotate4dY(r) * vec4(v - CENTER_3D, 1.)).xyz + CENTER_3D;
    #else
    return (rotate4dY(r) * vec4(v, 1.)).xyz;
    #endif
}
#endif
