#include "lab2xyz.glsl"
#include "xyz2rgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a Lab color to RGB color space.
    https://en.wikipedia.org/wiki/CIELAB_color_space
use: lab2rgb(<vec3|vec4> color)
*/

#ifndef FNC_LAB2RGB
#define FNC_LAB2RGB
vec3 lab2rgb(const in vec3 lab) { return xyz2rgb( lab2xyz( lab ) ); }
vec4 lab2rgb(const in vec4 lab) { return vec4(lab2rgb(lab.rgb), lab.a); }
#endif