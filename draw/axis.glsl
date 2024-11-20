
#include "line.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Draw the three axis (X,Y,Z) of a 3D object projected in 2D. The thickness of the axis can be adjusted.
use: <vec4> axis(<vec2> st, <mat4> M, <vec3> pos, <float> thickness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_AXIS
#define FNC_AXIS

vec4 axis(in vec2 st, mat4 M, vec3 pos, float thickness) {
    vec4 rta = vec4(0.0);

    vec4 center = M * vec4(pos, 1.0);
    center.xy /= center.w;
    center.xy = (center.xy * 0.5 + 0.5);

    vec4 axis[3];
    axis[0] = vec4(1.0, 0.0, 0.0, 1.0);
    axis[1] = vec4(0.0, 1.0, 0.0, 1.0);
    axis[2] = vec4(0.0, 0.0, 1.0, 1.0);

    for (int i = 0; i < 3; i++) {
        #ifdef DEBUG_FLIPPED_SPACE
        vec4 a = M * (vec4(pos - axis[i].xyz, 1.0));
        #else
        vec4 a = M * (vec4(pos + axis[i].xyz, 1.0));
        #endif
        a.xy /= a.w;
        a.xy = (a.xy * 0.5 + 0.5);
        rta += axis[i] * line(st, center.xy, a.xy, thickness);
    } 
     
    return rta;
}

#endif