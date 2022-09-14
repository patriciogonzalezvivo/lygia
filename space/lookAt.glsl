/*
original_author: Patricio Gonzalez Vivo
description: create a look up matrix 
use: 
    - <mat3> lookAt(<vec3> forward, <vec3> up)
    - <mat3> lookAt(<vec3> target, <vec3> eye, <vec3> up)
    - <mat3> lookAt(<vec3> target, <vec3> eye, <float> rolle)
*/

#ifndef FNC_LOOKAT
#define FNC_LOOKAT

mat3 lookAt(vec3 forward, vec3 up) {
    vec3 xaxis = normalize(cross(forward, up));
    vec3 yaxis = up;
    vec3 zaxis = forward;
    return mat3(xaxis, yaxis, zaxis);
}

mat3 lookAt(vec3 target, vec3 eye, vec3 up) {
    vec3 zaxis = normalize(target - eye);
    vec3 xaxis = normalize(cross(zaxis, up));
    vec3 yaxis = cross(zaxis, xaxis);
    return mat3(xaxis, yaxis, zaxis);
}

mat3 lookAt(vec3 target, vec3 eye, float roll) {
    vec3 up = vec3(sin(roll), cos(roll), 0.0);
    vec3 zaxis = normalize(target - eye);
    vec3 xaxis = normalize(cross(zaxis, up));
    vec3 yaxis = normalize(cross(xaxis, zaxis));
    return mat3(xaxis, yaxis, zaxis);
}

#endif