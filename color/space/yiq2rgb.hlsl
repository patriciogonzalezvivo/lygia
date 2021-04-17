/*
author: Patricio Gonzalez Vivo
description: pass a color in YIQ and get RGB color. From https://en.wikipedia.org/wiki/YIQ
use: yiq2rgb(<float3|float4> color)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_YIQ2RGB
#define FNC_YIQ2RGB

const float3x3 yiq2rgb_mat = float3x3(
    1.,     1.,     1.,
    .956,  -.272, -1.106,
    .621,  -.647,  1.703
);

float3 yiq2rgb(in float3 yiq) {
    return mul(yiq2rgb_mat, yiq);
}

float4 yiq2rgb(in float4 yiq) {
    return float4(yiq2rgb(yiq.rgb), yiq.a);
}
#endif
