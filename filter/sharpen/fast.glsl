/*
author: Johan Ismael
description: sharpening convolutional operation
use: sharpen(<sampler2D> texture, <vec2> st, <vec2> pixel)
options:
    SHARPENFAST_KERNELSIZE: Defaults 2
    SHARPENFAST_TYPE: defaults to vec3
    SHARPENFAST_SAMPLER_FNC(POS_UV): defaults to texture2D(tex, POS_UV).rgb
license: |
    Copyright (c) 2017 Johan Ismael
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef SHARPENFAST_KERNELSIZE
#ifdef SHARPEN_KERNELSIZE
#define SHARPENFAST_KERNELSIZE SHARPEN_KERNELSIZE
#else
#define SHARPENFAST_KERNELSIZE 2
#endif
#endif

#ifndef SHARPENFAST_TYPE
#ifdef SHARPEN_TYPE
#define SHARPENFAST_TYPE SHARPEN_TYPE
#else
#define SHARPENFAST_TYPE vec3
#endif
#endif

#ifndef SHARPENFAST_SAMPLER_FNC
#ifdef SHARPEN_SAMPLER_FNC
#define SHARPENFAST_SAMPLER_FNC(POS_UV) SHARPEN_SAMPLER_FNC(POS_UV)
#else
#define SHARPENFAST_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV).rgb
#endif
#endif

#ifndef FNC_SHARPENFAST
#define FNC_SHARPENFAST
SHARPENFAST_TYPE sharpenFast(in sampler2D tex, in vec2 coords, in vec2 pixel, float strenght) {
    SHARPENFAST_TYPE sum = SHARPENFAST_TYPE(0.);
    for (int i = 0; i < SHARPENFAST_KERNELSIZE; i++) {
        float f_size = float(i) + 1.;
        f_size *= strenght;
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + vec2( -1., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + vec2( 0., -1.) * pixel * f_size);
        sum +=  5. * SHARPENFAST_SAMPLER_FNC(coords + vec2( 0., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + vec2( 0., 1.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + vec2( 1., 0.) * pixel * f_size);
    }
    return sum / float(SHARPENFAST_KERNELSIZE);
}

SHARPENFAST_TYPE sharpenFast(in sampler2D tex, in vec2 coords, in vec2 pixel) {
    SHARPENFAST_TYPE sum = SHARPENFAST_TYPE(0.);
    for (int i = 0; i < SHARPENFAST_KERNELSIZE; i++) {
        float f_size = float(i) + 1.;
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + vec2( -1., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + vec2( 0., -1.) * pixel * f_size);
        sum +=  5. * SHARPENFAST_SAMPLER_FNC(coords + vec2( 0., 0.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + vec2( 0., 1.) * pixel * f_size);
        sum += -1. * SHARPENFAST_SAMPLER_FNC(coords + vec2( 1., 0.) * pixel * f_size);
    }
    return sum / float(SHARPENFAST_KERNELSIZE);
}
#endif