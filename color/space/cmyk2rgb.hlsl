/*
contributors: Patricio Gonzalez Vivo
description: convert CMYK to RGB
use: cmyk2rgb(<float 4> cmyk)
*/

#ifndef FNC_CMYK2RGB
#define FNC_CMYK2RGB
float3 cmyk2rgb(float4 cmyk) {
    float invK = 1.0 - cmyk.w;
    return saturate(1.0-min(float3(1.0, 1.0, 1.0), cmyk.xyz * invK + cmyk.w));
}
#endif