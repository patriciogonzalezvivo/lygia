/*
description: Computes the luminance of the specified linear RGB color using the luminance coefficients from Rec. 709.
use: luminance(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LUMINANCE
#define FNC_LUMINANCE
float luminance(in float3 _linear) { return dot(_linear, float3(0.2126, 0.7152, 0.0722)); }
float luminance(in float4 _linear) { return luminance( _linear.rgb ); }
#endif