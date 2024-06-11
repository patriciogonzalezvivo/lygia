#include "hsv2rgb.hlsl"
#include "rgb2ryb.hlsl"
#include "ryb2rgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Convert from HSV to RYB color space
use: <float3> hsv2ryb(<float3> hsv)
options:
    HSV2RYB_FAST: it will use a faster approximation of the conversion
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HSV2RYB
#define FNC_HSV2RYB
float3 hsv2ryb( in float3 v ) {
    #ifndef HSV2RYB_FAST
    float3 rgb = hsv2rgb(v);
    
    #ifdef RYB_FAST
    return ryb2rgb(rgb);
    #else
    return ryb2rgb(rgb) - saturate(1.-v.z);
    #endif

    #else
    float f = frac(v.x) * 6.0;
    float3 c = smoothstep(float3(3.,0.,3.), float3(2.,2.,4.), float3(f, f, f));
    c += smoothstep(float3(4.,3.,4.), float3(6.,4.,6.), float3(f, f, f)) * float3(1., -1., -1.);
    return lerp(float3(1., 1., 1.), c, v.y) * v.z;
    #endif
}
#endif