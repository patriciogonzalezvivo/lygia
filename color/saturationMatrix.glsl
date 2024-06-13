/*
contributors: Patricio Gonzalez Vivo
description: Generate a matrix to change a the saturation of any color
use: saturationMatrix(<float> amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SATURATIONMATRIX
#define FNC_SATURATIONMATRIX
mat4 saturationMatrix(in float a) {
    vec3 lum = vec3(.3086, .6094, .0820 );
    float iA= 1. - a;
    vec3 r = vec3(lum.x * iA) + vec3(a, .0, .0);
    vec3 g = vec3(lum.y * iA) + vec3( .0, a, .0);
    vec3 b = vec3(lum.z * iA) + vec3( .0, .0, a);
    return mat4(r,.0,
                g,.0,
                b,.0,
                .0, .0, .0, 1.);
}
#endif