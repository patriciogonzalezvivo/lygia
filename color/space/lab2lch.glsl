/*
contributors: Patricio Gonzalez Vivo
description: "Converts a LCh to Lab color space. \nNote: LCh is simply Lab but converted to polar coordinates (in degrees).\n"
use: <vec3|vec4> lab2rgb(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LAB2LCH
#define FNC_LAB2LCH
vec3 lab2lch(vec3 lab) {
    return vec3(
        lab.x,
        sqrt(dot(lab.yz, lab.yz)),
        atan(lab.z, lab.y) * 57.2957795131
    );
}
vec4 lab2lch(vec4 lab) { return vec4(lab2lch(lab.xyz), lab.a); }
#endif