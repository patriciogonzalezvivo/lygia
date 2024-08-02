/*
contributors:  Shadi El Hajj
description: Create a view matrix from camera position and camera rotation (euler angles)
use: <mat4> lookAtViewMatrix(in <vec3> position, in <vec3> euler)
*/

#include "../math/rotate3dX.glsl"
#include "../math/rotate3dY.glsl"
#include "../math/rotate3dZ.glsl"

#ifndef FNC_VIEWMATRIX
#define FNC_VIEWMATRIX

 mat4 viewMatrix(vec3 position, vec3 euler) {
    mat3 rotZ = rotate3dZ(euler.z);
    mat3 rotX = rotate3dX(euler.x);
    mat3 rotY = rotate3dY(euler.y);
    mat3 idendity = mat3(1.0);
    mat4 m = mat4(rotY * rotX * rotZ * idendity);
    m[0][3] = position.x;
    m[1][3] = position.y;
    m[2][3] = position.z;
    m[3][3] = 1.0;
    return m;
}

#endif