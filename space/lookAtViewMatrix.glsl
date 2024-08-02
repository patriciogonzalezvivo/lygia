/*
contributors:  Shadi El Hajj
description: Create a look-at view matrix
use: <mat4> lookAtViewMatrix(in <vec3> position, in <vec3> target, in <vec3> up)
*/

#include "lookAt.glsl"
#include "translate.glsl"

#ifndef FNC_LOOKATVIEWMATRIX
#define FNC_LOOKATVIEWMATRIX

mat4 lookAtViewMatrix( in vec3 position, in vec3 target, in vec3 up ) {
    mat4 m = mat4(lookAt(position, target, up));
    return translate(m, position);
}

mat4 lookAtViewMatrix( in vec3 position, in vec3 target, in float roll ) {
    mat3 m = lookAt(position, target, roll);
    return translate(m, position);
}

mat4 lookAtViewMatrix( in vec3 position, in vec3 lookAt ) {
    return lookAtViewMatrix(position, lookAt, vec3(0.0, 1.0, 0.0));
}

#endif