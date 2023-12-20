/*
contributors: Patricio Gonzalez Vivo
description: |
   Convert a color in YIQ to linear RGB color. 
   From https://en.wikipedia.org/wiki/YIQQ
use: <float3|float4> yiq2rgb(<float3|float4> color)
*/

#ifndef MAT_YIQ2RGB
#define MAT_YIQ2RGB
const float3x3 YIQ2RGB = float3x3(  1.0,  0.9469,  0.6235, 
                                    1.0, -0.2747, -0.6357, 
                                    1.0, -1.1085,  1.7020 );
#endif

#ifndef FNC_YIQ2RGB
#define FNC_YIQ2RGB
float3 yiq2rgb(in float3 yiq) { return mul(yiq2rgb_mat, yiq); }
float4 yiq2rgb(in float4 yiq) { return float4(yiq2rgb(yiq.rgb), yiq.a); }
#endif
