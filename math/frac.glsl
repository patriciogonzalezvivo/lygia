/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the floor function for float, vec2, vec33, and float4 types, to match HLSL's behavior.
use: <float|vec2|vec3|vec4> frac(<float|vec2|vec3|vec4> X)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_FRAC
#define FNC_FRAC
#define frac(X) fract(X)
#endif