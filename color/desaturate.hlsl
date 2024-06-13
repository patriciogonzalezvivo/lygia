/*
contributors: Patricio Gonzalez Vivo
description: Change saturation of a color
use: desaturate(<float|float3|float4> color, float amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_DESATURATE
#define FNC_DESATURATE
float3 desaturate(in float3 color, in float amount ) {
    float l = dot(float3(.3, .59, .11), color);
    return lerp(color, float3(l, l, l), amount);
}

float4 desaturate(in float4 color, in float amount ) {
    return float4(desaturate(color.rgb, amount), color.a);
}
#endif
