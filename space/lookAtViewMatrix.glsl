/*
contributors:  Shadi El Hajj
description: Create a look-at view matrix
use: <mat4> lookAtViewMatrix(in <vec3> position, in <vec3> target, in <vec3> up)
*/

#include "lookAt.glsl"

#ifndef FNC_LOOKATVIEWMATRIX
#define FNC_LOOKATVIEWMATRIX

mat4 lookAtViewMatrix( in vec3 position, in vec3 target, in vec3 up ) {
    mat4 m = mat4(lookAt(position, target, up));
    m[0][3] = position.x;
    m[1][3] = position.y;
    m[2][3] = position.z;
    m[3][3] = 1.0;
    return m;
}

mat4 lookAtViewMatrix( in vec3 position, in vec3 target, in float roll ) {
    mat4 m = mat4(lookAt(position, target, roll));
    m[0][3] = position.x;
    m[1][3] = position.y;
    m[2][3] = position.z;
    m[3][3] = 1.0;
    return m;
}

mat4 lookAtViewMatrix( in vec3 position, in vec3 lookAt ) {
    return lookAtViewMatrix(position, lookAt, vec3(0.0, 1.0, 0.0));
}

#endif