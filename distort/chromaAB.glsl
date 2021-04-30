#include "../math/lengthSq.glsl"

/*
author: Patricio Gonzalez Vivo, Johan Ismael
description: Chroma Aberration inspired by https://www.shadertoy.com/view/4sX3z4
use: chromaAB(<sampler2D> texture, <vec2> st [, <float|vec2> sdf|offset, <float> pct])
options:
    CHROMAAB_TYPE: return type, defauls to vec3
    CHROMAAB_PCT: amount of aberration, defaults to 1.5
    CHROMAAB_SAMPLER_FNC: function used to sample the input texture, defaults to texture2D(tex, POS_UV)
    CHROMAAB_CENTER_BUFFER: scalar to attenuate the sdf passed in   
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo and Johan Ismael
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef CHROMAAB_PCT
#define CHROMAAB_PCT 1.5
#endif

#ifndef CHROMAAB_TYPE
#define CHROMAAB_TYPE vec3
#endif

#ifndef CHROMAAB_SAMPLER_FNC
#define CHROMAAB_SAMPLER_FNC(POS_UV) texture2D(tex, POS_UV)
#endif

#ifndef FNC_CHROMAAB
#define FNC_CHROMAAB

CHROMAAB_TYPE chromaAB(in sampler2D tex, in vec2 st, in vec2 direction, in vec3 distortion ) {
    vec2 offset = vec2(0.0);
    CHROMAAB_TYPE c = CHROMAAB_TYPE(1.);
    c.r = CHROMAAB_SAMPLER_FNC(st + direction * distortion.r).r;
    c.g = CHROMAAB_SAMPLER_FNC(st + direction * distortion.g).g;
    c.b = CHROMAAB_SAMPLER_FNC(st + direction * distortion.b).b;
    return c;
}

CHROMAAB_TYPE chromaAB(in sampler2D tex, in vec2 st, in vec2 offset, in float pct) {

  #ifdef CHROMAAB_CENTER_BUFFER
    // modify the distance from the center, so that only the edges are affected
    offset = max(offset - CHROMAAB_CENTER_BUFFER, 0.);
  #endif

  // Distort the UVs
  vec2 stR = st * (1.0 + offset * 0.02 * pct),
       stB = st * (1.0 - offset * 0.02 * pct);

  // Get the individual channels using the modified UVs
  CHROMAAB_TYPE c = CHROMAAB_TYPE(1.);
  c.r = CHROMAAB_SAMPLER_FNC(stR).r;
  c.g = CHROMAAB_SAMPLER_FNC(st).g;
  c.b = CHROMAAB_SAMPLER_FNC(stB).b;
  return c;
}

CHROMAAB_TYPE chromaAB(in sampler2D tex, in vec2 st, in float sdf, in float pct) {
  return chromaAB(tex, st, vec2(sdf), pct);
}

CHROMAAB_TYPE chromaAB(in sampler2D tex, in vec2 st, in float sdf) {
  return chromaAB(tex, st, sdf, CHROMAAB_PCT);
}

CHROMAAB_TYPE chromaAB(in sampler2D tex, in vec2 st, in vec2 offset) {
  return chromaAB(tex, st, offset, CHROMAAB_PCT);
}

CHROMAAB_TYPE chromaAB(in sampler2D tex, in vec2 st) {
  return chromaAB(tex, st, lengthSq(st - .5), CHROMAAB_PCT);
}

#endif
