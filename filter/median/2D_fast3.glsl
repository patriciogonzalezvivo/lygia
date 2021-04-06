/*
author: [Morgan McGuire, Kyle Whitson]
description: |
  3x3 median filter, adapted from "A Fast, Small-Radius GPU Median Filter" 
  by Morgan McGuire in ShaderX6 https://casual-effects.com/research/McGuire2008Median/index.html
use: median2D_fast3(<sampler2D> texture, <vec2> st, <vec2> pixel)
options:
  MEDIAN2D_FAST3_TYPE: default vec4
  MEDIAN2D_FAST3_SAMPLER_FNC(POS_UV): default texture2D(tex, POS_UV)
license:
  Copyright (c) Morgan McGuire and Williams College, 2006. All rights reserved.
  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef MEDIAN2D_FAST3_TYPE
#define MEDIAN2D_FAST3_TYPE vec4
#endif

#ifndef MEDIAN2D_FAST3_SAMPLER_FNC
#define MEDIAN2D_FAST3_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV)
#endif

#ifndef MEDIAN_S2
#define MEDIAN_S2(a, b) temp = a; a = min(a, b); b = max(temp, b);
#endif

#ifndef MEDIAN_2
#define MEDIAN_2(a, b) MEDIAN_S2(v[a], v[b]);
#endif

#ifndef FNC_MEDIAN2D_FAST3
#define FNC_MEDIAN2D_FAST3
#define MEDIAN_MN3(a, b, c) MEDIAN_2(a, b); MEDIAN_2(a, c);
#define MEDIAN_MX3(a, b, c) MEDIAN_2(b, c); MEDIAN_2(a, c);
#define MEDIAN_MNMX3(a, b, c) MEDIAN_MX3(a, b, c); MEDIAN_2(a, b);                                                                // 3 exchanges
#define MEDIAN_MNMX4(a, b, c, d) MEDIAN_2(a, b); MEDIAN_2(c, d); MEDIAN_2(a, c); MEDIAN_2(b, d);                                  // 4 exchanges
#define MEDIAN_MNMX5(a, b, c, d, e) MEDIAN_2(a, b); MEDIAN_2(c, d); MEDIAN_MN3(a, c, e); MEDIAN_MX3(b, d, e);                     // 6 exchanges
#define MEDIAN_MNMX6(a, b, c, d, e, f) MEDIAN_2(a, d); MEDIAN_2(b, e); MEDIAN_2(c, f); MEDIAN_MN3(a, b, c); MEDIAN_MX3(d, e, f);  // 7 exchanges
MEDIAN2D_FAST3_TYPE median2D_fast3(in sampler2D tex, in vec2 st, in vec2 radius) {
  MEDIAN2D_FAST3_TYPE v[9];
  for (int dX = -1; dX <= 1; ++dX) {
      for (int dY = -1; dY <= 1; ++dY) {
          vec2 offset = vec2(float(dX), float(dY));
          // If a pixel in the window is located at (x+dX, y+dY), put it at index (dX + R)(2R + 1) + (dY + R) of the
          // pixel array. This will fill the pixel array, with the top left pixel of the window at pixel[0] and the
          // bottom right pixel of the window at pixel[N-1].
          v[(dX + 1) * 3 + (dY + 1)] = MEDIAN2D_FAST3_SAMPLER_FNC(st + offset * radius);
      }
  }
  MEDIAN2D_FAST3_TYPE temp = MEDIAN2D_FAST3_TYPE(0.);
  MEDIAN_MNMX6(0, 1, 2, 3, 4, 5);
  MEDIAN_MNMX5(1, 2, 3, 4, 6);
  MEDIAN_MNMX4(2, 3, 4, 7);
  MEDIAN_MNMX3(3, 4, 8);
  return v[4];
}
#endif
