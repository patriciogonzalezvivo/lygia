#include "lab2xyz.glsl"
#include "xyz2rgb.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Converts a Lab color to RGB color space.
use: lab2rgb(<vec3|vec4> color)
*/

#ifndef FNC_LAB2RGB
#define FNC_LAB2RGB
vec3 lab2rgb(in vec3 lab) {
    return xyz2rgb( lab2xyz( vec3(100. * lab.x,
                                    2. * 127. * (lab.y - .5),
                                    2. * 127. * (lab.z - .5)) ) );
}

vec4 lab2rgb(in vec4 lab) { return vec4(lab2rgb(lab.rgb), lab.a); }
#endif