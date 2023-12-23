#include "lch2lab.glsl"
#include "lab2rgb.glsl"
/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a Lch to linear RGB color space. 
    Note: LCh is simply Lab but converted to polar coordinates (in degrees).
use: <vec3|vec4> lch2rgb(<vec3|vec4> color)
*/

#ifndef FNC_LCH2RGB
#define FNC_LCH2RGB
vec3 lch2rgb(vec3 lch) { return lab2rgb( lch2lab(lch) ); }
vec4 lch2rgb(vec4 lch) { return vec4(lch2rgb(lch.xyz),lch.a);}
#endif