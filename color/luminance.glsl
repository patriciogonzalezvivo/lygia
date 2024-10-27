/*
contributor: nan
description: Computes the luminance of the specified linear RGB color using the luminance coefficients from Rec. 709.
use: luminance(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LUMINANCE 
#define FNC_LUMINANCE
float luminance(vec3 v) { return dot(v, vec3(0.21250175, 0.71537574, 0.07212251)); }
float luminance(vec4 v) { return luminance( v.rgb ); }
#endif
