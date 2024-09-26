/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Destination Over Compositing
use: <float4> compositeDestinationOver(<float4> src, <float4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_COMPOSITE_DST_OVER
#define FNC_COMPOSITE_DST_OVER

float compositeDestinationOver(float src, float dst) {
    return dst + src * (1.0 - dst);
}

float3 compositeDestinationOver(float3 srcColor, float3 dstColor, float srcAlpha, float dstAlpha) {
    return dstColor + srcColor * (1.0 - dstAlpha);
}

float4 compositeDestinationOver(float4 srcColor, float4 dstColor) {
    float4 result;

    result.rgb = compositeDestinationOver(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeDestinationOver(srcColor.a, dstColor.a);

    return result;
}
#endif
