#include "lch2lab.glsl"
#include "lab2srgb.glsl"
/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a Lch to sRGB color space. 
    Note: LCh is simply Lab but converted to polar coordinates (in degrees).
use: lch2srgb(<vec3|vec4> color)
*/

#ifndef FNC_LCH2SRGB
#define FNC_LCH2SRGB
vec3 lch2srgb(vec3 lch) { return lab2srgb( lch2lab(lch) ); }
vec4 lch2srgb(vec4 lch) { return vec4(lch2srgb(lch.xyz),lch.a);}
#endif