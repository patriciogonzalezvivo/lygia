#include "hsv2rgb.hlsl"
#include "rgb2ryb.hlsl"
#include "ryb2rgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Convert from HSV to RYB color space
use: <float3> hsv2ryb(<float3> hsv)
options:
    HSV2RGB_CMY_BIAS: if this is defined, the function will use the CMY bias version of the HSV2RGB function
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HSV2RYB
#define FNC_HSV2RYB
float3 hsv2ryb( in float3 v ) {
    #ifdef HSV2RGB_CMY_BIAS
    float3 rgb = hsv2rgb(v);
    return ryb2rgb(rgb);

    #else
    float f = frac(v.x) * 6.0;
    float3 c = smoothstep(float3(3.,0.,3.), float3(2.,2.,4.), float3(f, f, f));
    c += smoothstep(float3(4.,3.,4.), float3(6.,4.,6.), float3(f, f, f)) * float3(1., -1., -1.);
    return lerp(float3(1., 1., 1.), c, v.y) * v.z;
    #endif
}
#endif