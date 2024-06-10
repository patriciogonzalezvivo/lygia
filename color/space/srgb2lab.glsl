#include "rgb2lab.glsl"
#include "srgb2rgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a sRGB color to Lab
use: <vec3|vec4> srgb2lab(<vec3|vec4> rgb)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SRGB2LAB
#define FNC_SRGB2LAB
vec3 srgb2lab(const in vec3 srgb) { return rgb2lab(srgb2rgb(srgb));}
vec4 srgb2lab(const in vec4 srgb) { return vec4(srgb2lab(srgb.rgb),rgb.a); }
#endif
