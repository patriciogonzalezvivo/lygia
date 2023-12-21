#include "lab2xyz.glsl"
#include "xyz2srgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a Lab color to RGB color space.
    https://en.wikipedia.org/wiki/CIELAB_color_space
use: <vec3|vec4> lab2srgb(<vec3|vec4> color)
*/

#ifndef FNC_LAB2SRGB
#define FNC_LAB2SRGB
vec3 lab2srgb(const in vec3 lab) { return xyz2srgb( lab2xyz( lab ) ); }
vec4 lab2srgb(const in vec4 lab) { return vec4(lab2srgb(lab.rgb), lab.a); }
#endif