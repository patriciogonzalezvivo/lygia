/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 scale matrix
use:
    - <mat4> scale4d(<float|vec3|vec4> radians)
    - <mat4> scale4d(<float> x, <float> y, <float> z [, <float> w])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SCALE4D
mat4 scale4d(float s) {
    return mat4(s, 0.0, 0.0, 0.0,
                0.0, s, 0.0, 0.0,
                0.0, 0.0, s, 0.0,
                0.0, 0.0, 0.0, 1.0 );
}

mat4 scale4d(float x, float y, float z) {
    return mat4( x, 0.0, 0.0, 0.0,
                0.0,  y, 0.0, 0.0,
                0.0, 0.0,  z, 0.0,
                0.0, 0.0, 0.0, 1.0);
}

mat4 scale4d(float x, float y, float z, float w) {
    return mat4( x, 0.0, 0.0, 0.0,
                0.0,  y, 0.0, 0.0,
                0.0, 0.0,  z, 0.0,
                0.0, 0.0, 0.0,  w );
}

mat4 scale4d(vec3 s) {
    return mat4(s.x, 0.0, 0.0, 0.0,
                0.0, s.y, 0.0, 0.0,
                0.0, 0.0, s.z, 0.0,
                0.0, 0.0, 0.0, 1.0 );
}

mat4 scale4d(vec4 s) {
    return mat4(s.x, 0.0, 0.0, 0.0,
                0.0, s.y, 0.0, 0.0,
                0.0, 0.0, s.z, 0.0,
                0.0, 0.0, 0.0, s.w );
}
#endif
