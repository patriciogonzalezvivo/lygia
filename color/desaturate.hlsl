/*
contributors: Patricio Gonzalez Vivo
description: change saturation of a color
use: desaturate(<float|float3|float4> color, float amount)
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
