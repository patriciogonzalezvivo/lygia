#include "rgb2lch.glsl"
#include "srgb2rgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a sRGB color to Lab
use: <vec3|vec4> srgb2lch(<vec3|vec4> rgb)
*/

#ifndef FNC_SRGB2LCH
#define FNC_SRGB2LCH
vec3 srgb2lch(const in vec3 srgb) { return rgb2lch(srgb2rgb(srgb));}
vec4 srgb2lch(const in vec4 srgb) { return vec4(srgb2lch(srgb.rgb),srgb.a); }
#endif
