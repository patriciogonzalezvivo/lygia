/*
author: Patricio Gonzalez Vivo
description: two dimension Gaussian Blur to be applied in only one passes
use: gaussianBlur2D(<sampler2D> texture, <vec2> st, <vec2> pixel_direction , const int kernelSize)
options:
    GAUSSIANBLUR2D_TYPE: Default `vec4`
    GAUSSIANBLUR2D_SAMPLER_FNC(POS_UV): Default `texture2D(tex, POS_UV)`
    GAUSSIANBLUR2D_KERNELSIZE: Use only for WebGL 1.0 and OpenGL ES 2.0 . For example RaspberryPis is not happy with dynamic loops. Default is 'kernelSize'
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

#ifndef GAUSSIANBLUR2D_TYPE
#ifdef GAUSSIANBLUR_TYPE
#define GAUSSIANBLUR2D_TYPE GAUSSIANBLUR_TYPE
#else
#define GAUSSIANBLUR2D_TYPE vec4
#endif
#endif

#ifndef GAUSSIANBLUR2D_SAMPLER_FNC
#ifdef GAUSSIANBLUR_SAMPLER_FNC
#define GAUSSIANBLUR2D_SAMPLER_FNC(POS_UV) GAUSSIANBLUR_SAMPLER_FNC(POS_UV)
#else
#define GAUSSIANBLUR2D_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV)
#endif
#endif

#ifndef FNC_GAUSSIANBLUR2D
#define FNC_GAUSSIANBLUR2D
GAUSSIANBLUR2D_TYPE gaussianBlur2D(in sampler2D tex, in vec2 st, in vec2 offset, const int kernelSize) {
    GAUSSIANBLUR2D_TYPE accumColor = GAUSSIANBLUR2D_TYPE(0.);
    #ifndef GAUSSIANBLUR2D_KERNELSIZE
    #define GAUSSIANBLUR2D_KERNELSIZE kernelSize
    #endif

    float accumWeight = 0.;
    const float k = .15915494; // 1 / (2*PI)
    float kernelSize2 = float(GAUSSIANBLUR2D_KERNELSIZE) * float(GAUSSIANBLUR2D_KERNELSIZE);

    for (int j = 0; j < GAUSSIANBLUR2D_KERNELSIZE; j++) {
        float y = -.5 * (float(GAUSSIANBLUR2D_KERNELSIZE) - 1.) + float(j);
        for (int i = 0; i < GAUSSIANBLUR2D_KERNELSIZE; i++) {
            float x = -.5 * (float(GAUSSIANBLUR2D_KERNELSIZE) - 1.) + float(i);
            float weight = (k / float(GAUSSIANBLUR2D_KERNELSIZE)) * exp(-(x * x + y * y) / (2. * kernelSize2));
            GAUSSIANBLUR2D_TYPE tex = GAUSSIANBLUR2D_SAMPLER_FNC(st + vec2(x, y) * offset);
            accumColor += weight * tex;
            accumWeight += weight;
        }
    }
    return accumColor / accumWeight;
}
#endif
