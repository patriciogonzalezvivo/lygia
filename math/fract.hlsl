/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the floor function for float2, float3, and float4 types, to match GLSL's behavior.
use: <float2|float3|float4> fract(<float2|float3|float4> value);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_FRACT
#define FNC_FRACT
#define fract(X) frac(X)
#endif