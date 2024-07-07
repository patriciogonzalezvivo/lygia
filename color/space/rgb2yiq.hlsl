/*
contributors: Patricio Gonzalez Vivo
description: "Convert from linear RGB to YIQ which was the followin range. \nUsing conversion matrices from FCC NTSC Standard (SMPTE C) https://en.wikipedia.org/wiki/YIQ\n"
use: <float3|float4> rgb2yiq(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MAT_RGB2YIQ
#define MAT_RGB2YIQ
static const float3x3 RGB2YIQ = float3x3(
    0.300,  0.5900,  0.1100, 
    0.599, -0.2773, -0.3217, 
    0.213, -0.5251,  0.3121);
#endif

#ifndef FNC_RGB2YIQ
#define FNC_RGB2YIQ
float3 rgb2yiq(in float3 rgb) { return mul(RGB2YIQ, rgb); }
float4 rgb2yiq(in float4 rgb) { return float4(rgb2yiq(rgb.rgb), rgb.a); }
#endif
