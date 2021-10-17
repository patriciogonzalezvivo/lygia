#include "../color/space/YCbCr2rgb.glsl"

/*
author: Patricio Gonzalez Vivo
description: a way to solve the YUV camare texture on iOS
use: textureYUV(<sampler2D> tex1, <sampler2D> tex2, <vec2> st)
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_TEXTUREYUV
#define FNC_TEXTUREYUV
vec4 textureYUV(in sampler2D tex_Y, in sampler2D tex_UV, in vec2 st) {
  vec4 yuv = vec4(0.);
  yuv.xw = texture2D(tex_Y, st).ra;
  yuv.yz = texture2D(tex_UV, st).rg;
  return YCbCr2rgb(yuv);
}
#endif
