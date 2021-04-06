/*
author: Patricio Gonzalez Vivo
description: simple two dimentional box blur, so can be apply in a single pass
use: boxBlur1D_fast9(<sampler2D> texture, <vec2> st, <vec2> pixel_direction)
options:
  BOXBLUR2D_FAST9_TYPE: Default is `vec4`
  BOXBLUR2D_FAST9_SAMPLER_FNC(POS_UV): Default is `texture2D(tex, POS_UV)`
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef BOXBLUR2D_FAST9_TYPE
#define BOXBLUR2D_FAST9_TYPE vec4
#endif

#ifndef BOXBLUR2D_FAST9_SAMPLER_FNC
#define BOXBLUR2D_FAST9_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV)
#endif

#ifndef FNC_BOXBLUR2D_FAST9
#define FNC_BOXBLUR2D_FAST9
BOXBLUR2D_FAST9_TYPE boxBlur2D_fast9(in sampler2D tex, in vec2 st, in vec2 offset) {
  BOXBLUR2D_FAST9_TYPE color = BOXBLUR2D_FAST9_SAMPLER_FNC(st);           // center
  color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(-offset.x, offset.y));  // tleft
  color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(-offset.x, 0.));        // left
  color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(-offset.x, -offset.y)); // bleft
  color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(0., offset.y));         // top
  color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(0., -offset.y));        // bottom
  color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + offset);                     // tright
  color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(offset.x, 0.));         // right
  color += BOXBLUR2D_FAST9_SAMPLER_FNC(st + vec2(offset.x, -offset.y));  // bright
  return color * 0.1111111111; // 1./9.
}
#endif
