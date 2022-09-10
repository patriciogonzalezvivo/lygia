/*
author: Patricio Gonzalez Vivo
description: simple one dimentional box blur, to be applied in two passes
use: boxBlur1D(<sampler2D> texture, <vec2> st, <vec2> pixel_offset, <int> kernelSize)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BOXBLUR1D_TYPE: default is vec4
    - BOXBLUR1D_SAMPLER_FNC(POS_UV): default texture2D(tex, POS_UV)
    - BOXBLUR1D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef BOXBLUR1D_TYPE
#ifdef BOXBLUR_TYPE
#define BOXBLUR1D_TYPE BOXBLUR_TYPE
#else
#define BOXBLUR1D_TYPE vec4
#endif
#endif

#ifndef BOXBLUR1D_SAMPLER_FNC
#ifdef BOXBLUR_SAMPLER_FNC
#define BOXBLUR1D_SAMPLER_FNC(POS_UV) BOXBLUR_SAMPLER_FNC(POS_UV)
#else
#define BOXBLUR1D_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV)
#endif
#endif

#ifndef FNC_BOXBLUR1D
#define FNC_BOXBLUR1D
BOXBLUR1D_TYPE boxBlur1D(in sampler2D tex, in vec2 st, in vec2 offset, const int kernelSize) {
    BOXBLUR1D_TYPE color = BOXBLUR1D_TYPE(0.);
    #ifndef BOXBLUR1D_KERNELSIZE
    #define BOXBLUR1D_KERNELSIZE kernelSize
    #endif

    float f_kernelSize = float(BOXBLUR1D_KERNELSIZE);
    float weight = 1. / f_kernelSize;

    for (int i = 0; i < BOXBLUR1D_KERNELSIZE; i++) {
        float x = -.5 * (f_kernelSize - 1.) + float(i);
        color += BOXBLUR1D_SAMPLER_FNC(st + offset * x ) * weight;
    }
    return color;
}
#endif
