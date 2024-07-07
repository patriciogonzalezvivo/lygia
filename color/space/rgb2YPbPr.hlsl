/*
contributors: Patricio Gonzalez Vivo
description: Pass a color in RGB and get it in YPbPr from http://www.equasys.de/colorconversion.html
use: <float3|float4> rgb2YPbPr(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MAT_RGB2PBPR
#define MAT_RGB2PBPR
#ifdef YPBPR_SDTV
static const float3x3 RGB2PBPR = float3x3( 
    .299, -.169,  .5,
    .587, -.331, -.419,
    .114,  .5,   -.081
);
#else
static const float3x3 RGB2PBPR = float3x3( 
    0.2126, -0.1145721060573399,   0.5,
    0.7152, -0.3854278939426601,  -0.4541529083058166,
    0.0722,  0.5,                 -0.0458470916941834
);
#endif
#endif

#ifndef FNC_RGB2YPBPR
#define FNC_RGB2YPBPR
float3 rgb2YPbPr(in float3 rgb) { return mul(RGB2PBPR, rgb); }
float4 rgb2YPbPr(in float4 rgb) { return float4(rgb2YPbPr(rgb.rgb),rgb.a); }
#endif
