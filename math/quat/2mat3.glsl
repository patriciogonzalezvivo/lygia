#include "type.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: given a quaternion, returns a rotation 3x3 matrix
use: <mat3> quat2mat3(<QUAT> Q)
*/

#ifndef FNC_QUAT2MAT3
#define FNC_QUAT2MAT3

mat3 quat2mat3(QUAT q) {
    float qxx = q.x * q.x;
    float qyy = q.y * q.y;
    float qzz = q.z * q.z;
    float qxz = q.x * q.z;
    float qxy = q.x * q.y;
    float qyw = q.y * q.w;
    float qzw = q.z * q.w;
    float qyz = q.y * q.z;
    float qxw = q.x * q.w;

    return mat3(
        vec3(1.0 - 2.0 * (qyy + qzz), 2.0 * (qxy - qzw), 2.0 * (qxz + qyw)),
        vec3(2.0 * (qxy + qzw), 1.0 - 2.0 * (qxx + qzz), 2.0 * (qyz - qxw)),
        vec3(2.0 * (qxz - qyw), 2.0 * (qyz + qxw), 1.0 - 2.0 * (qxx + qyy))
    );
}
#endif