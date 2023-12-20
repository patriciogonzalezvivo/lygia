#include "srgb2xyz.glsl"
#include "rgb2srgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB color to XYZ color space.
use: <vec3|vec4> rgb2xyz(<vec3|vec4> rgb)
*/

#ifndef FNC_RGB2XYZ
#define FNC_RGB2XYZ
vec3 rgb2xyz(in vec3 rgb) { return SRGB2XYZ * rgb2srgb(rgb);}
vec4 rgb2xyz(in vec4 rgb) { return vec4(rgb2xyz(rgb.rgb),rgb.a); }
#endif