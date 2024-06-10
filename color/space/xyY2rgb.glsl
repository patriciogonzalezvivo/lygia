#include "xyz2rgb.glsl"
#include "xyY2xyz.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Converts from xyY to linear RGB'
use: <vec3|vec4> xyY2rgb(<vec3|vec4> xyY)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_XYY2RGB
#define FNC_XYY2RGB
vec3 xyY2rgb(const in vec3 xyY) { return xyz2rgb(xyY2xyz(xyY));}
vec4 xyY2rgb(const in vec4 xyY) { return vec4(xyz2rgb(xyY2xyz(xyY.xyz)), xyY.w);}
#endif
