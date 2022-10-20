#include "../math/rotate4dZ.glsl"

/*
function: rotateZ
original_author: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateZ(<vec3|vec4> pos, float radian [, vec3 center])
options:
  - CENTER_3D
*/

#ifndef FNC_ROTATEZ
#define FNC_ROTATEZ
vec4 rotateZ(in vec4 pos, in float radian, in vec4 center) {
    return rotate4dZ(radian) * (pos - center) + center;
}

vec4 rotateZ(in vec4 pos, in float radian) {
    #ifdef CENTER_4D
    return rotateZ(pos, radian, CENTER_3D);
    #else
    return rotateZ(pos, radian, vec4(.0));
    #endif
}

vec3 rotateZ(in vec3 pos, in float radian, in vec3 center) {
    return (rotate4dZ(radian) * vec4(pos - center, 0.) ).xyz + center;
}

vec3 rotateZ(in vec3 pos, in float radian) {
    #ifdef CENTER_3D
    return rotateZ(pos, radian, CENTER_3D);
    #else
    return rotateZ(pos, radian, vec3(.0));
    #endif
}
#endif
