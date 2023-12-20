#include "rgb2xyz.glsl"
#include "xyz2xyY.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a linear RGB color to xyY color space.
use: <vec3|vec4> rgb2xyY(<vec3|vec4> rgb)
*/

#ifndef FNC_RGB2XYY
#define FNC_RGB2XYY
vec3 rgb2xyY(vec3 rgb) { return xyz2xyY(rgb2xyz(rgb));}
vec4 rgb2xyY(vec4 rgb) { return vec4(rgb2xyY(rgb.rgb), rgb.a);}
#endif