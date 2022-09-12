/*
author: Patricio Gonzalez Vivo
description: sharpening filter
use: sharpen(<sampler2D> texture, <vec2> st, <vec2> renderSize [, float streanght])
options:
    - SHARPEN_KERNELSIZE: Defaults 2
    - SHARPEN_TYPE: defaults to vec3
    - SHARPEN_SAMPLER_FNC(POS_UV): defaults to texture2D(tex, POS_UV).rgb
    - SHARPEN_FNC: defaults to sharpenFast
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef RADIALBLUR_SAMPLER_FNC
#define RADIALBLUR_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV)
#endif

#ifndef SHARPEN_TYPE
#define SHARPEN_TYPE vec3
#endif

#ifndef SHARPEN_SAMPLER_FNC
#define SHARPEN_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV).rgb
#endif

#ifndef SHARPEN_FNC
#define SHARPEN_FNC sharpenFast
#endif

#include "sharpen/fast.glsl"
#include "sharpen/adaptive.glsl"
#include "sharpen/contrastAdaptive.glsl"

#ifndef FNC_SHARPEN
#define FNC_SHARPEN

SHARPEN_TYPE sharpen(in sampler2D tex, in vec2 st, in vec2 pixel, float strenght) {
    return SHARPEN_FNC (tex, st, pixel, strenght);
}

SHARPEN_TYPE sharpen(in sampler2D tex, in vec2 st, in vec2 pixel) {
    return SHARPEN_FNC (tex, st, pixel);
}

#endif 