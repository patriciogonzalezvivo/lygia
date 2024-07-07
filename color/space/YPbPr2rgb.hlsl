/*
contributors: Patricio Gonzalez Vivo
description: Pass a color in RGB and get it in YPbPr from http://www.equasys.de/colorconversion.html
use: YPbPr2RGB(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MAT_YPBPR2RGB
#define MAT_YPBPR2RGB
#ifdef YPBPR_SDTV
static const float3x3 YPBPR2RGB = float3x3( 
    1.0,     1.0,       1.0,
    0.0,    -0.344,     1.772,
    1.402,  -0.714,     0.0
);
#else
static const float3x3 YPBPR2RGB = float3x3( 
    1.0,     1.0,       1.0,
    0.0,    -0.187,     1.856,
    1.575,  -0.468,     0.0
);
#endif
#endif

#ifndef FNC_YPBPR2RGB
#define FNC_YPBPR2RGB
float3 YPbPr2rgb(in float3 rgb) { return mul(YPBPR2RGB, rgb); }
float4 YPbPr2rgb(in float4 rgb) { return float4(YPbPr2rgb(rgb.rgb),rgb.a); }
#endif