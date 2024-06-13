/*
contributors: Patricio Gonzalez Vivo
description: Bias high pass
use: <float4|float3|float> contrast(<float4|float3|float> value, <float> amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_CONTRAST
#define FNC_CONTRAST
float contrast(in float value, in float amount) {
    return (value - 0.5 ) * amount + 0.5;
}

float3 contrast(in float3 value, in float amount) {
    return (value - 0.5 ) * amount + 0.5;
}

float4 contrast(in float4 value, in float amount) {
    return float4(contrast(value.rgb, amount), value.a);
}
#endif
