#include "xyz2rgb.glsl"
#include "rgb2srgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a XYZ color to sRGB. From http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html'
use: xyz2srgb(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_XYZ2SRGB
#define FNC_XYZ2SRGB
vec3 xyz2srgb(const in vec3 xyz) { return rgb2srgb(xyz2rgb(xyz)); }
vec4 xyz2srgb(const in vec4 xyz) { return vec4(xyz2srgb(xyz.rgb), xyz.a); }
#endif