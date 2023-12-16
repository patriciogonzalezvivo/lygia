/*
contributors: Patricio Gonzalez Vivo
description: create a perspective matrix
use: <mat4> perspective(<float> fov, <float> aspect, <float> near, <float> far);
*/

#ifndef FNC_PERSPECTIVE
#define FNC_PERSPECTIVE

mat4 perspective(float fov, float aspect, float near, float far) {
    float f = 1.0 / tan(fov / 2.0);
    float nf = 1.0 / (near - far);
    return mat4(
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) * nf, -1.0,
        0.0, 0.0, (2.0 * far * near) * nf, 0.0
    );
}

#endif