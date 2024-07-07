/*
contributors: Patricio Gonzalez Vivo
description: Pass a color in YUB and get RGB color
use: yuv2rgb(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef MAT_YUV2RGB
#define MAT_YUV2RGB
#ifdef YUV_SDTV
static const float3x3 YUV2RGB = float3x3(
    1.0,       1.0,      1.0,
    0.0,      -0.39465,  2.03211,
    1.13983,  -0.58060,  0.0
);
#else
static const float3x3 YUV2RGB = float3x3(
    1.0,       1.0,      1.0,
    0.0,      -0.21482,  2.12798,
    1.28033,  -0.38059,  0.0
);
#endif
#endif

#ifndef FNC_YUV2RGB
#define FNC_YUV2RGB
float3 yuv2rgb(in float3 yuv) { return mul(YUV2RGB, yuv); }
float4 yuv2rgb(in float4 yuv) { return float4(yuv2rgb(yuv.rgb), yuv.a); }
#endif
