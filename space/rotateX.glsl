#include "../math/rotate4dX.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateX(<vec3|vec4> pos, float radian [, vec3 center])
options:
    - CENTER_3D
*/

#ifndef FNC_ROTATEX
#define FNC_ROTATEX
vec4 rotateX(in vec4 pos, in float radian, in vec4 center) {
    return rotate4dX(radian) * (pos - center) + center;
}

vec4 rotateX(in vec4 pos, in float radian) {
    #ifdef CENTER_4D
    return rotateX(pos, radian, CENTER_4D);
    #else
    return rotateX(pos, radian, vec4(.0));
    #endif
}

vec3 rotateX(in vec3 pos, in float radian, in vec3 center) {
    return (rotate4dX(radian) * vec4(pos - center, 1.)).xyz + center;
}

vec3 rotateX(in vec3 pos, in float radian) {
    #ifdef CENTER_3D
    return rotateX(pos, radian, CENTER_3D);
    #else
    return rotateX(pos, radian, vec3(.0));
    #endif
}
#endif
