/*
original_author: Patricio Gonzalez Vivo
description: pass a color in RGB and get it in YPbPr from http://www.equasys.de/colorconversion.html
use: rgb2YPbPr(<float3|vec4> color)
*/

#ifndef FNC_RGB2YPBPR
#define FNC_RGB2YPBPR

#ifdef YPBPR_SDTV
const float3x3 rgb2YPbPr_mat = float3x3( 
    .299, -.169,  .5,
    .587, -.331, -.419,
    .114,  .5,   -.081
);
#else
const float3x3 rgb2YPbPr_mat = float3x3( 
    0.2126, -0.1145721060573399,   0.5,
    0.7152, -0.3854278939426601,  -0.4541529083058166,
    0.0722,  0.5,                 -0.0458470916941834
);
#endif

float3 rgb2YPbPr(in float3 rgb) {
    return mul(rgb2YPbPr_mat, rgb);
}

float4 rgb2YPbPr(in float4 rgb) {
    return float4(rgb2YPbPr(rgb.rgb),rgb.a);
}
#endif
