/*
contributors: Patricio Gonzalez Vivo
description: Change the exposure of a color
use: exposure(<float|float3|float4> color, float amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_EXPOSURE
#define FNC_EXPOSURE
float exposure(float value, float amount) {
    return value * pow(2., amount);
}

float3 exposure(float3 color, float amount) {
    return color * pow(2., amount);
}

float4 exposure(float4 color, float amount) {
    return float4(exposure( color.rgb, amount ), color.a);
}
#endif