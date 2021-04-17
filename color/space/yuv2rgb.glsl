/*
author: Patricio Gonzalez Vivo
description: pass a color in YUB and get RGB color
use: yuv2rgb(<vec3|vec4> color)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_YUV2RGB
#define FNC_YUV2RGB

#ifdef YUV_SDTV
const mat3 yuv2rgb_mat = mat3(
    1.,       1. ,      1.,
    0.,       -.39465,  2.03211,
    1.13983,  -.58060,  0.
);
#else
const mat3 yuv2rgb_mat = mat3(
    1.,       1. ,      1.,
    0.,       -.21482,  2.12798,
    1.28033,  -.38059,  0.
);
#endif

vec3 yuv2rgb(in vec3 yuv) {
    return yuv2rgb_mat * yuv;
}

vec4 yuv2rgb(in vec4 yuv) {
    return vec4(yuv2rgb(yuv.rgb), yuv.a);
}
#endif
