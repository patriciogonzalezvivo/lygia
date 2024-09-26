/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Source Out Compositing
use: <vec4> compositeSourceOut(<vec4> src, <vec4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_COMPOSITE_SRC_OUT
#define FNC_COMPOSITE_SRC_OUT

float compositeSourceOut(float src, float dst) {
    return src * (1.0 - dst);
}

vec3 compositeSourceOut(vec3 srcColor, vec3 dstColor, float srcAlpha, float dstAlpha) {
    return srcColor * (1.0 - dstAlpha);
}

vec4 compositeSourceOut(vec4 srcColor, vec4 dstColor) {
    vec4 result = vec4(0.0);

    result.rgb = compositeSourceOut(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeSourceOut(srcColor.a, dstColor.a);

    return result;
}
#endif
