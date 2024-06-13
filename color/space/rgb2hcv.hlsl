/*
contributors: Patricio Gonzalez Vivo
description: 'Convert from linear RGB to HCV (Hue, Chroma, Value). Based on work by Sam Hocevar and Emil Persson'
use: <float3|float4> rgb2xyz(<float3|float4> rgb)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef HCV_EPSILON
#define HCV_EPSILON 1e-10
#endif

#ifndef FNC_RGB2HCV
#define FNC_RGB2HCV
float3 rgb2hcv(float3 rgb) {
    float4 P = (rgb.g < rgb.b) ? float4(rgb.bg, -1.0, 2.0/3.0) : float4(rgb.gb, 0.0, -1.0/3.0);
    float4 Q = (rgb.r < P.x) ? float4(P.xyw, rgb.r) : float4(rgb.r, P.yzx);
    float C = Q.x - min(Q.w, Q.y);
    float H = abs((Q.w - Q.y) / (6.0 * C + HCV_EPSILON) + Q.z);
    return float3(H, C, Q.x);
}
float4 rgb2hcv(float4 rgb) {return float4(rgb2hcv(rgb.rgb), rgb.a);}
#endif