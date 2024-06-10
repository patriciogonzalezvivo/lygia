#include "xyz2srgb.glsl"
#include "xyY2xyz.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Converts from xyY to sRGB'
use: <vec3|vec4> xyY2srgb(<vec3|vec4> xyY)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_XYY2SRGB
#define FNC_XYY2SRGB
vec3 xyY2srgb(const in vec3 xyY) { return xyz2srgb(xyY2xyz(xyY));}
vec4 xyY2srgb(const in vec4 xyY) { return vec4(xyY2srgb(xyY.xyz), xyY.w);}
#endif
