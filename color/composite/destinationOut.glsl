/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Destination Out Compositing
use: <vec4> compositeDestinationOut(<vec4> src, <vec4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_COMPOSITE_DST_OUT
#define FNC_COMPOSITE_DST_OUT

float compositeDestinationOut(float src, float dst) {
    return dst * (1.0 - src);
}

vec3 compositeDestinationOut(vec3 srcColor, vec3 dstColor, float srcAlpha, float dstAlpha) {
    return dstColor * (1.0 - srcAlpha);
}

vec4 compositeDestinationOut(vec4 srcColor, vec4 dstColor)  {
    vec4 result = vec4(0.0);
   
    result.rgb = compositeDestinationOut(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeDestinationOut(srcColor.a, dstColor.a);

    return result;
}
#endif
