/*
author: Patricio Gonzalez Vivo
description: pass a color in RGB and get it in YUB
use: rgb2yuv(<float3|float4> color)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_RGB2YUV
#define FNC_RGB2YUV

#ifdef YUV_SDTV
const float3x3 rgb2yuv_mat = float3x3(
    .299, -.14713,  .615,
    .587, -.28886, -.51499,
    .114,  .436,   -.10001
);
#else
const float3x3 rgb2yuv_mat = float3x3(
    .2126,  -.09991, .615,
    .7152,  -.33609,-.55861,
    .0722,   .426,  -.05639
);
#endif

float3 rgb2yuv(in float3 rgb) {
    return mul(rgb2yuv_mat, rgb);
}

float4 rgb2yuv(in float4 rgb) {
    return float4(rgb2yuv(rgb.rgb),rgb.a);
}
#endif
