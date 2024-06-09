#include "space/rgb2hsv.hlsl"
#include "space/hsv2rgb.hlsl"

/*
contributors: Johan Ismael
description: Shifts color hue
use: hueShift(<float3|float4> color, <float> amount)
license:
  - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
  - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HUESHIFT
#define FNC_HUESHIFT
float3 hueShift(in float3 color, in float amount) {
    float3 hsv = rgb2hsv(color);
    hsv.r += amount;
    return hsv2rgb(hsv);
}

float4 hueShift(in float4 color, in float amount) {
    return float4(hueShift(color.rgb, amount), color.a);
}
#endif