/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Destination In Compositing
use: <float4> compositeDestinationIn(<float4> src, <float4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_COMPOSITE_DST_IN
#define FNC_COMPOSITE_DST_IN

float compositeDestinationIn(float src, float dst) {
    return dst * src;
}

float3 compositeDestinationIn(float3 srcColor, float3 dstColor, float srcAlpha, float dstAlpha) {
    return dstColor * srcAlpha;
}


float4 compositeDestinationIn(float4 srcColor, float4 dstColor) {
    float4 result;
   
    result.rgb = compositeDestinationIn(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeDestinationIn(srcColor.a, dstColor.a);

    return result;
}
#endif
