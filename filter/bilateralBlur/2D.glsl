#include "../../color/space/rgb2luma.glsl"

/*
author: Patricio Gonzalez Vivo
description: two dimensional bilateral Blur, to do it in one single pass
use: bilateralBlur2D(<sampler2D> texture, <vec2> st, <vec2> offset, <int> kernelSize)
options:
    - BILATERALBLUR2D_TYPE: default is vec3
    - BILATERALBLUR2D_SAMPLER_FNC(POS_UV): default texture2D(tex, POS_UV)
    - BILATERALBLUR2D_LUMA(RGB): default rgb2luma
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef BILATERALBLUR2D_TYPE
#ifdef BILATERALBLUR_TYPE
#define BILATERALBLUR2D_TYPE BILATERALBLUR_TYPE
#else
#define BILATERALBLUR2D_TYPE vec4
#endif
#endif

#ifndef BILATERALBLUR2D_SAMPLER_FNC
#ifdef BILATERALBLUR_SAMPLER_FNC
#define BILATERALBLUR2D_SAMPLER_FNC(POS_UV) BILATERALBLUR_SAMPLER_FNC(POS_UV)
#else
#define BILATERALBLUR2D_SAMPLER_FNC(POS_UV) SAMPLER_FNC(tex, POS_UV)
#endif
#endif

#ifndef BILATERALBLUR2D_LUMA
#define BILATERALBLUR2D_LUMA(RGB) rgb2luma(RGB.rgb)
#endif

#ifndef FNC_BILATERALBLUR2D
#define FNC_BILATERALBLUR2D
BILATERALBLUR2D_TYPE bilateralBlur2D(in sampler2D tex, in vec2 st, in vec2 offset, const int kernelSize) {
  BILATERALBLUR2D_TYPE accumColor = BILATERALBLUR2D_TYPE(0.);
  #ifndef BILATERALBLUR2D_KERNELSIZE
  #define BILATERALBLUR2D_KERNELSIZE kernelSize
  #endif
  float accumWeight = 0.;
  const float k = .15915494; // 1. / (2.*PI)
  const float k2 = k * k;
  float kernelSize2 = float(BILATERALBLUR2D_KERNELSIZE) * float(BILATERALBLUR2D_KERNELSIZE);
  BILATERALBLUR2D_TYPE tex0 = BILATERALBLUR2D_SAMPLER_FNC(st);
  float lum0 = BILATERALBLUR2D_LUMA(tex0);

  for (int j = 0; j < BILATERALBLUR2D_KERNELSIZE; j++) {
    float dy = -.5 * (float(BILATERALBLUR2D_KERNELSIZE) - 1.0) + float(j);
    for (int i = 0; i < BILATERALBLUR2D_KERNELSIZE; i++) {
      float dx = -.5 * (float(BILATERALBLUR2D_KERNELSIZE) - 1.0) + float(i);
      BILATERALBLUR2D_TYPE tex = BILATERALBLUR2D_SAMPLER_FNC(st + vec2(dx, dy) * offset);
      float lum = BILATERALBLUR2D_LUMA(tex);
      float dl = 255. * (lum - lum0);
      float weight = (k2 / kernelSize2) * exp(-(dx * dx + dy * dy + dl * dl) / (2. * kernelSize2));
      accumColor += weight * tex;
      accumWeight += weight;
    }
  }
  return accumColor / accumWeight;
}
#endif
