#include "rgb2xyz.glsl"
#include "xyz2lab.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB color to Lab color space.
use: <vec3|vec4> rgb2lab(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RGB2LAB
#define FNC_RGB2LAB
vec3 rgb2lab(const in vec3 c) { return xyz2lab( rgb2xyz( c ) ); }
vec4 rgb2lab(const in vec4 rgb) { return vec4(rgb2lab(rgb.rgb),rgb.a); }
#endif