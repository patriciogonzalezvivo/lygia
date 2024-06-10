#include "hue2rgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a HCY color to linear RGB'
use: <float3|float4> hcy2rgb(<float3|float4> hsl)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HCY2RGB
#define FNC_HCY2RGB
float3 hcy2rgb(float3 hcy) {
    const float3 HCYwts = float3(0.299, 0.587, 0.114);
    float3 RGB = hue2rgb(hcy.x);
    float Z = dot(RGB, HCYwts);
    if (hcy.z < Z) {
        hcy.y *= hcy.z / Z;
    } else if (Z < 1.0) {
        hcy.y *= (1.0 - hcy.z) / (1.0 - Z);
    }
    return (RGB - Z) * hcy.y + hcy.z;
}
float4 hcy2rgb(float4 hcy) { return float4(hcy2rgb(hcy.rgb), hcy.a); }
#endif