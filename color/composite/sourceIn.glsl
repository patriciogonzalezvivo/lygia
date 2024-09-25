/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Source In Compositing
use: <vec4> compositeSourceIn(<vec4> src, <vec4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_COMPOSITE_SRC_IN
#define FNC_COMPOSITE_SRC_IN

float compositeSourceIn(float src, float dst) {
    return src * dst;
}

vec3 compositeSourceIn(vec3 srcColor, vec3 dstColor, float srcAlpha, float dstAlpha) {
    return srcColor * dstAlpha;
}

vec4 compositeSourceIn(vec4 srcColor, vec4 dstColor) {
    vec4 result = vec4(0.0);

    result.rgb = compositeSourceIn(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeSourceIn(srcColor.a, dstColor.a);

    return result;
}

#endif
