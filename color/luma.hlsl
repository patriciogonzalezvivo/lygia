#include "space/rgb2luma.hlsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Get the luminosity of a color. From https://github.com/hughsk/glsl-luma/blob/master/index.hlsl
use: luma(<float3|float4> color)
*/

#ifndef FNC_LUMA
#define FNC_LUMA
float luma(in float color) {
    return float(color);
}

float luma(in float3 color) {
    return rgb2luma(color);
}

float luma(in float4 color) {
    return rgb2luma(color.rgb);
}
#endif
