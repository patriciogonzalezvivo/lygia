#include "rgb2lab.glsl"
#include "lab2lch.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB color to LCh color space.
use: <vec3|vec4> rgb2lch(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RGB2LCH
#define FNC_RGB2LCH
vec3 rgb2lch(const in vec3 rgb) { return lab2lch(rgb2lab(rgb)); }
vec4 rgb2lch(const in vec4 rgb) { return vec4(rgb2lch(rgb.rgb),rgb.a); }
#endif