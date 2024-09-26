/*
contributors: Patricio Gonzalez Vivo, Anton Marini
description: Porter Duff Source Over Compositing
use: <vec4> compositeSourceOver(<vec4> src, <vec4> dst)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_COMPOSITE_SRC_OVER
#define FNC_COMPOSITE_SRC_OVER

float compositeSourceOver(float src, float dst) {
    return src + dst * (1.0 - src);
}

vec3 compositeSourceOver(vec3 srcColor, vec3 dstColor, float srcAlpha, float dstAlpha) {
    return srcColor * srcAlpha + dstColor * dstAlpha * (1.0 - srcAlpha);
}

vec4 compositeSourceOver(vec4 srcColor, vec4 dstColor) {
    vec4 result = vec4(0.0);
   
    result.rgb = compositeSourceOver(srcColor.rgb, dstColor.rgb, srcColor.a, dstColor.a);
    result.a = compositeSourceOver(srcColor.a, dstColor.a);

    return result;
}



#endif
