/*
original_author: Hugh Kennedy (https://github.com/hughsk)
description: get's the luminosity of a color. From https://github.com/hughsk/glsl-luma/blob/master/index.glsl
use: rgb2luma(<float3 float4> color)
*/

#ifndef FNC_RGB2LUMA
#define FNC_RGB2LUMA
float rgb2luma(in float3 color) {
    return dot(color, float3(0.299, 0.587, 0.114));
}

float rgb2luma(in float4 color) {
    return rgb2luma(color.rgb);
}
#endif
