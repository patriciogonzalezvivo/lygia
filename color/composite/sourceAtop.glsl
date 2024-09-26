/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Source Atop Compositing
use: <vec4> compositeSourceAtop(<vec4> src, <vec4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_COMPOSITE_SRC_ATOP
#define FNC_COMPOSITE_SRC_ATOP

float compositeSourceAtop(float src, float dst) {
    return src * dst + dst * (1.0 - src);
}

vec3 compositeSourceAtop(vec3 srcColor, vec3 dstColor, float srcAlpha, float dstAlpha) {
    return srcColor * dstAlpha + dstColor * (1.0 - srcAlpha);
}

vec4 compositeSourceAtop(vec4 srcColor, vec4 dstColor) {
    vec4 result = vec4(0.0);

    result.rgb = compositeSourceAtop(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeSourceAtop(srcColor.a, dstColor.a);

    return result;
}
#endif
