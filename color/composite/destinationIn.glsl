/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Destination In Compositing
use: <vec4> compositeDestinationIn(<vec4> src, <vec4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_COMPOSITE_DST_IN
#define FNC_COMPOSITE_DST_IN

float compositeDestinationIn(float src, float dst) {
    return dst * src;
}

vec3 compositeDestinationIn(vec3 srcColor, vec3 dstColor, float srcAlpha, float dstAlpha) {
    return dstColor * srcAlpha;
}

vec4 compositeDestinationIn(vec4 srcColor, vec4 dstColor) {
    vec4 result = vec4(0.0);
   
    result.rgb = compositeDestinationIn(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeDestinationIn(srcColor.a, dstColor.a);

    return result;
}
#endif
