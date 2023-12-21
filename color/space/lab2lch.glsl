/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a LCh to Lab color space. 
    Note: LCh is simply Lab but converted to polar coordinates (in degrees).
use: <vec3|vec4> lab2rgb(<vec3|vec4> color)
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