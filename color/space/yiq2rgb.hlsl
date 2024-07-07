/*
contributors: Patricio Gonzalez Vivo
description: "Converts a color in YIQ to linear RGB color. \nFrom https://en.wikipedia.org/wiki/YIQQ\n"
use: <float3|float4> yiq2rgb(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MAT_YIQ2RGB
#define MAT_YIQ2RGB
static const float3x3 YIQ2RGB = float3x3(  1.0,  0.9469,  0.6235, 
                                    1.0, -0.2747, -0.6357, 
                                    1.0, -1.1085,  1.7020 );
#endif

#ifndef FNC_YIQ2RGB
#define FNC_YIQ2RGB
float3 yiq2rgb(in float3 yiq) { return mul(YIQ2RGB, yiq); }
float4 yiq2rgb(in float4 yiq) { return float4(yiq2rgb(yiq.rgb), yiq.a); }
#endif
