/*
author: Patricio Gonzalez Vivo
description: make a radial blur, with dir as the direction to the center and strength as the amount
use: radialBlur(<sampler2D> texture, <vec2> st, <vec2> dir [, <float> strength] )
options:
  RADIALBLUR_KERNELSIZE: Default 64 
  RADIALBLUR_STRENGTH: Default 0.125
  RADIALBLUR_TYPE: Default `vec4`
  RADIALBLUR_SAMPLER_FNC(POS_UV): Default `texture2D(tex, POS_UV)`
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef RADIALBLUR_KERNELSIZE
#define RADIALBLUR_KERNELSIZE 64
#endif

#ifndef RADIALBLUR_STRENGTH
#define RADIALBLUR_STRENGTH .125
#endif

#ifndef RADIALBLUR_TYPE
#define RADIALBLUR_TYPE vec4
#endif

#ifndef RADIALBLUR_SAMPLER_FNC
#define RADIALBLUR_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV)
#endif

#ifndef FNC_RADIALBLUR
#define FNC_RADIALBLUR
RADIALBLUR_TYPE radialBlur(in sampler2D tex, in vec2 st, in vec2 dir, in float strength) {
    RADIALBLUR_TYPE color = RADIALBLUR_TYPE(0.);
    float f_samples = float(RADIALBLUR_KERNELSIZE);
    float f_factor = 1./f_samples;
    for (int i = 0; i < RADIALBLUR_KERNELSIZE; i += 2) {
        color += RADIALBLUR_SAMPLER_FNC(st + float(i) * f_factor * dir * strength);
        color += RADIALBLUR_SAMPLER_FNC(st + float(i+1) * f_factor * dir * strength);
    }
    return color * f_factor;
}

RADIALBLUR_TYPE radialBlur(in sampler2D tex, in vec2 st, in vec2 dir) {
    return radialBlur(tex, st, dir, RADIALBLUR_STRENGTH);
}
#endif
