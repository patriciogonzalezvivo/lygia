#include "../../math/mmin.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: convert CMYK to RGB
use: rgb2cmyk(<float3|float4> rgba)
*/

#ifndef FNC_RGB2CMYK
#define FNC_RGB2CMYK
float4 rgb2cmyk(float3 rgb) {
    float k = mmin(1.0 - rgb);
    float invK = 1.0 - k;
    float3 cmy = (1.0 - rgb - k) / invK;
    cmy *= step(0.0, invK);
    return saturate(float4(cmy, k));
}
#endif