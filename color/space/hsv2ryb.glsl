#include "hsv2rgb.glsl"
#include "rgb2ryb.glsl"
#include "ryb2rgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Convert from HSV to RYB color space
use: <vec3> hsv2ryb(<vec3> hsv)
options:
    HSV2RYB_FAST: if this is defined, the function will use the CMY bias version of the HSV2RGB function
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HSV2RYB
#define FNC_HSV2RYB
vec3 hsv2ryb( in vec3 v ) {
    #ifndef HSV2RYB_FAST
    vec3 rgb = hsv2rgb(v);

    #ifdef RYB_FAST
    return ryb2rgb(rgb);
    #else
    return ryb2rgb(rgb) - saturate(1.-v.z);
    #endif

    #else
    float f = fract(v.x) * 6.0;
    vec3 c = smoothstep(vec3(3.,0.,3.), vec3(2.,2.,4.), vec3(f));
    c += smoothstep(vec3(4.,3.,4.), vec3(6.,4.,6.), vec3(f)) * vec3(1., -1., -1.);
    return mix(vec3(1.), c, v.y) * v.z;
    #endif
}
#endif