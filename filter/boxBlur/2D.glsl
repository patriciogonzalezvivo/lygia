/*
author: Patricio Gonzalez Vivo
description: simple two dimentional box blur, so can be apply in a single pass
use: boxBlur2D(<sampler2D> texture, <vec2> st, <vec2> pixel_offset, <int> kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BOXBLUR2D_TYPE: Default `vec4`
    - BOXBLUR2D_SAMPLER_FNC(POS_UV): default is `texture2D(tex, POS_UV)`
    - BOXBLUR2D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef BOXBLUR2D_TYPE
#ifdef BOXBLUR_TYPE
#define BOXBLUR2D_TYPE BOXBLUR_TYPE
#else
#define BOXBLUR2D_TYPE vec4
#endif
#endif

#ifndef BOXBLUR2D_SAMPLER_FNC
#ifdef BOXBLUR_SAMPLER_FNC
#define BOXBLUR2D_SAMPLER_FNC(POS_UV) BOXBLUR_SAMPLER_FNC(POS_UV)
#else
#define BOXBLUR2D_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV)
#endif
#endif

#ifndef FNC_BOXBLUR2D
#define FNC_BOXBLUR2D
BOXBLUR2D_TYPE boxBlur2D(in sampler2D tex, in vec2 st, in vec2 pixel, const int kernelSize) {
    BOXBLUR2D_TYPE color = BOXBLUR2D_TYPE(0.);
    #ifndef BOXBLUR2D_KERNELSIZE
    #define BOXBLUR2D_KERNELSIZE kernelSize
    #endif

    float accumWeight = 0.;
    float f_kernelSize = float(BOXBLUR2D_KERNELSIZE);
    float kernelSize2 = f_kernelSize * f_kernelSize;
    float weight = 1. / kernelSize2;

    for (int j = 0; j < BOXBLUR2D_KERNELSIZE; j++) {
        float y = -.5 * (f_kernelSize - 1.) + float(j);
        for (int i = 0; i < BOXBLUR2D_KERNELSIZE; i++) {
            float x = -.5 * (f_kernelSize - 1.) + float(i);
            color += BOXBLUR2D_SAMPLER_FNC(st + vec2(x, y) * pixel) * weight;
        }
    }
    return color;
}
#endif
