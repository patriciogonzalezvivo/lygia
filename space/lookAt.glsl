/*
contributors: Patricio Gonzalez Vivo
description: create a look at matrix. Right handed by default.
use:
    - <mat3> lookAt(<vec3> forward, <vec3> up)
    - <mat3> lookAt(<vec3> eye, <vec3> target, <vec3> up)
    - <mat3> lookAt(<vec3> eye, <vec3> target, <float> roll)
    - <mat3> lookAt(<vec3> forward)
options:
    - LOOK_AT_LEFT_HANDED: assume a left-handed coordinate system
    - LOOK_AT_RIGHT_HANDED: assume a right-handed coordinate system
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LOOKAT
#define FNC_LOOKAT

mat3 lookAt(vec3 forward, vec3 up) {
    vec3 zaxis = normalize(forward);
#if defined(LOOK_AT_RIGHT_HANDED)
    vec3 xaxis = normalize(cross(zaxis, up));
    vec3 yaxis = cross(xaxis, zaxis);
#else
    vec3 xaxis = normalize(cross(up, zaxis));
    vec3 yaxis = cross(zaxis, xaxis);
#endif
    return mat3(xaxis, yaxis, zaxis);
}

mat3 lookAt(vec3 eye, vec3 target, vec3 up) {
    vec3 forward = normalize(target - eye);
    return lookAt(forward, up);
}

mat3 lookAt(vec3 eye, vec3 target, float roll) {
    vec3 up = vec3(sin(roll), cos(roll), 0.0);
    return lookAt(eye, target, up);
}

mat3 lookAt(vec3 forward) {
    return lookAt(forward, vec3(0.0, 1.0, 0.0));
}

#endif