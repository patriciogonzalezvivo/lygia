#include "../math/rotate4dY.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateY(<vec3> pos, float radian [, vec4 center])
options:
    - CENTER_3D
*/

#ifndef FNC_ROTATEY
#define FNC_ROTATEY
vec4 rotateY(in vec4 pos, in float radian, in vec4 center) {
    return rotate4dY(radian) * (pos - center) + center;
}

vec4 rotateY(in vec4 pos, in float radian) {
    #ifdef CENTER_4D
    return rotateY(pos, radian, CENTER_4D);
    #else
    return rotateY(pos, radian, vec4(.0));
    #endif
}

vec3 rotateY(in vec3 pos, in float radian, in vec3 center) {
    return (rotate4dY(radian) * vec4((pos - center), 1.)).xyz + center;
}

vec3 rotateY(in vec3 pos, in float radian) {
    #ifdef CENTER_3D
    return rotateY(pos, radian, CENTER_3D);
    #else
    return rotateY(pos, radian, vec3(.0));
    #endif
}
#endif
