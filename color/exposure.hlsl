/*
contributors: Patricio Gonzalez Vivo
description: change the exposure of a color
use: exposure(<float|float3|float4> color, float amount)
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