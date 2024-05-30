#include "rgb2xyz.glsl"
#include "srgb2rgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a sRGB color to XYZ
use: <vec3|vec4> srgb2xyz(<vec3|vec4> srgb)
*/

#ifndef FNC_SRGB2XYZ
#define FNC_SRGB2XYZ
vec3 srgb2xyz(const in vec3 srgb) { return rgb2xyz(srgb2rgb(srgb));}
vec4 srgb2xyz(const in vec4 srgb) { return vec4(srgb2xyz(srgb.rgb),srgb.a); }
#endif
