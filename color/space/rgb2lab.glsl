#include "rgb2xyz.glsl"
#include "xyz2lab.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Converts a RGB color to Lab color space.
use: rgb2lab(<vec3|vec4> color)
*/

#ifndef FNC_RGB2LAB
#define FNC_RGB2LAB
vec3 rgb2lab(in vec3 c) {
    vec3 lab = xyz2lab( rgb2xyz( c ) );
    return vec3(lab.x / 100.0,
                0.5 + 0.5 * (lab.y / 127.0),
                0.5 + 0.5 * (lab.z / 127.0));
}

vec4 rgb2lab(in vec4 rgb) { return vec4(rgb2lab(rgb.rgb),rgb.a); }
#endif