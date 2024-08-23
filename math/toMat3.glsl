/*
contributors: Patricio Gonzalez Vivo
description: given a 4x4 returns a 3x3
use: <mat4> toMat3(<mat3> m)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_TOMAT3
#define FNC_TOMAT3
mat3 toMat3(mat4 m) {
    #if __VERSION__ >= 300
    return mat3(m);
    #else
    return mat3(m[0].xyz, m[1].xyz, m[2].xyz);
    #endif
}
#endif