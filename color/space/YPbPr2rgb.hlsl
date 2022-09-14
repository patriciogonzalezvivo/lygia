/*
original_author: Patricio Gonzalez Vivo
description: pass a color in RGB and get it in YPbPr from http://www.equasys.de/colorconversion.html
use: YPbPr2RGB(<float3|vec4> color)
*/

#ifndef FNC_YPBPR2RGB
#define FNC_YPBPR2RGB

#ifdef YPBPR_SDTV
const float3x3 YPbPr2rgb_mat = float3x3( 
    1.,     1.,       1.,
    0.,     -.344,    1.772,
    1.402,  -.714,    0.
);
#else
const float3x3 YPbPr2rgb_mat = float3x3( 
    1.,     1.,       1.,
    0.,     -.187,    1.856,
    1.575,  -.468,    0.
);
#endif

float3 YPbPr2rgb(in float3 rgb) {
    return mul(YPbPr2rgb_mat, rgb);
}

float4 YPbPr2rgb(in float4 rgb) {
    return float4(YPbPr2rgb(rgb.rgb),rgb.a);
}
#endif