#include "lch2lab.glsl"
#include "lab2rgb.glsl"
/*
contributors: Patricio Gonzalez Vivo
description: "Converts a Lch to linear RGB color space. \nNote: LCh is simply Lab but converted to polar coordinates (in degrees).\n"
use: <vec3|vec4> lch2rgb(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LCH2RGB
#define FNC_LCH2RGB
vec3 lch2rgb(vec3 lch) { return lab2rgb( lch2lab(lch) ); }
vec4 lch2rgb(vec4 lch) { return vec4(lch2rgb(lch.xyz),lch.a);}
#endif