#include "rgb2hcv.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Convert from linear RGB to HSL. Based on work by Sam Hocevar and Emil Persson'
use: <float3|float4> rgb2hsl(<float3|float4> rgb)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef HSL_EPSILON
#define HSL_EPSILON 1e-10
#endif

#ifndef FNC_RGB2HSL
#define FNC_RGB2HSL
float3 rgb2hsl(float3 rgb) {
    float3 HCV = rgb2hcv(rgb);
    float L = HCV.z - HCV.y * 0.5;
    float S = HCV.y / (1.0 - abs(L * 2.0 - 1.0) + HSL_EPSILON);
    return float3(HCV.x, S, L);
}
float4 rgb2hsl(float4 rgb) { return float4(rgb2hsl(rgb.xyz),rgb.a);}
#endif