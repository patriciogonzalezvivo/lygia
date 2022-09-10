#include "../color/space/rgb2luma.glsl"

/*
author: Patricio Gonzalez Vivo
description: TODO
use: bilateralBlur(<sampler2D> texture, <vec2> st, <vec2> duv)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BILATERALBLUR_AMOUNT
    - BILATERALBLUR_TYPE
    - BILATERALBLUR_SAMPLER_FNC
license: |
    Copyright (c) 20212 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef BILATERALBLUR_AMOUNT
#define BILATERALBLUR_AMOUNT bilateralBlur13
#endif

#ifndef BILATERALBLUR_TYPE
#define BILATERALBLUR_TYPE vec4
#endif

#ifndef BILATERALBLUR_SAMPLER_FNC
#define BILATERALBLUR_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV)
#endif

#ifndef BILATERALBLUR_LUMA
#define BILATERALBLUR_LUMA(RGB) rgb2luma(RGB.rgb)
#endif

#include "bilateralBlur/2D.glsl"

#ifndef FNC_BILATERALFILTER
#define FNC_BILATERALFILTER
BILATERALBLUR_TYPE bilateralBlur(in sampler2D tex, in vec2 st, in vec2 offset, const int kernelSize) {
    return bilateralBlur2D(tex, st, offset, kernelSize);
}

BILATERALBLUR_TYPE bilateralBlur13(in sampler2D tex, in vec2 st, in vec2 offset) {
    return bilateralBlur(tex, st, offset, 7);
}

BILATERALBLUR_TYPE bilateralBlur9(in sampler2D tex, in vec2 st, in vec2 offset) {
    return bilateralBlur(tex, st, offset, 5);
}

BILATERALBLUR_TYPE bilateralBlur5(in sampler2D tex, in vec2 st, in vec2 offset) {
    return bilateralBlur(tex, st, offset, 3);
}

BILATERALBLUR_TYPE bilateralBlur(in sampler2D tex, in vec2 st, in vec2 offset) {
    return BILATERALBLUR_AMOUNT(tex, st, offset);
}
#endif
