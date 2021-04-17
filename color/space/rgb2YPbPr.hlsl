/*
author: Patricio Gonzalez Vivo
description: pass a color in RGB and get it in YPbPr from http://www.equasys.de/colorconversion.html
use: rgb2YPbPr(<float3|vec4> color)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
