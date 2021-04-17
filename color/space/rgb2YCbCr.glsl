/*
author: Patricio Gonzalez Vivo
description: convert RGB to YCbCr according to https://en.wikipedia.org/wiki/YCbCr
use: rgb2YCbCr(<vec3|vec4> color)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_RGB2YCBCR
#define FNC_RGB2YCBCR
vec3 rgb2YCbCr(in vec3 rgb){
    float y = dot(rgb, vec3(.299, .587, .114));
    float cb = .5 + dot(rgb, vec3(-.168736, -.331264, .5));
    float cr = .5 + dot(rgb, vec3(.5, -.418688, -.081312));
    return vec3(y, cb, cr);
}

vec4 rgb2YCbCr(in vec4 rgb) {
    return vec4(rgb2YCbCr(rgb.rgb),rgb.a);
}
#endif