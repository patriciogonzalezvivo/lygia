/*
contributors: Patricio Gonzalez Vivo
description: Generate a matrix to change a the saturation of any color
use: saturationMatrix(<float> amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SATURATIONMATRIX
#define FNC_SATURATIONMATRIX
float4x4 saturationMatrix(in float amount) {
    float3 lum = float3(.3086, .6094, .0820);
    float invAmount= 1. - amount;

    float3 red = float3(1.0, 1.0, 1.0) * lum.x * invAmount;
    red += float3(amount, .0, .0);

    float3 green = float3(1.0, 1.0, 1.0) * lum.y * invAmount;
    green += float3( .0, amount, .0);

    float3 blue = float3(1.0, 1.0, 1.0) * lum.z * invAmount;
    blue += float3( .0, .0, amount);

    return float4x4(red.x, green.x, blue.x, .0,
                    red.y, green.y, blue.y, .0,
                    red.z, green.z, blue.z, .0,
                    .0, .0, .0, 1.);
}
#endif
