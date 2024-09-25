/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Destination Atop Compositing
use: <float4> compositeDestinationAtop(<float4> src, <float4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_COMPOSITE_DST_ATOP
#define FNC_COMPOSITE_DST_ATOP

float compositeDestinationAtop(float src, float dst) {
    return dst * src + src * (1.0 - dst);
}

float3 compositeDestinationAtop(float3 srcColor, float3 dstColor, float srcAlpha, float dstAlpha) {
    return dstColor * srcAlpha + srcColor * (1.0 - dstAlpha);
}

float4 compositeDestinationAtop(float4 srcColor, float4 dstColor) {
    float4 result;
   
    result.rgb = compositeDestinationAtop(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeDestinationAtop(srcColor.a, dstColor.a);

    return result;
}
#endif
