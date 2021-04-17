/*
author: Patricio Gonzalez Vivo
description: convert YCbCr to RGB according to https://en.wikipedia.org/wiki/YCbCr
use: YCbCr2rgb(<vec3|vec4> color)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_YCBCR2RGB
#define FNC_YCBCR2RGB
vec3 YCbCr2rgb(in vec3 ycbcr) {
    float cb = ycbcr.y - .5;
    float cr = ycbcr.z - .5;
    float y = ycbcr.x;
    float r = 1.402 * cr;
    float g = -.344 * cb - .714 * cr;
    float b = 1.772 * cb;
    return vec3(r, g, b) + y;
}

vec4 YCbCr2rgb(in vec4 ycbcr) {
    return vec4(YCbCr2rgb(ycbcr.rgb),ycbcr.a);
}
#endif
