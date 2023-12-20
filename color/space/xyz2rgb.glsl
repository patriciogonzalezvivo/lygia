#include "xyz2srgb.glsl"
#include "srgb2rgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a XYZ color to linear RGB.
    From http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
use: xyz2rgb(<vec3|vec4> color)
*/

#ifndef FNC_XYZ2RGB
#define FNC_XYZ2RGB
vec3 xyz2rgb(const in vec3 xyz) { return srgb2rgb(xyz2srgb(xyz)); }
vec4 xyz2rgb(const in vec4 xyz) { return vec4(xyz2rgb(xyz.rgb), xyz.a); }
#endif