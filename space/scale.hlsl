/*
original_author: Patricio Gonzalez Vivo
description: scale a 2D space variable
use: scale(<float2> st, <float2|float> scale_factor [, <float2> center])
options:
    - CENTER
    - CENTER_2D
    - CENTER_3D
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_SCALE
#define FNC_SCALE
float scale(in float st, in float s, in float center) {
  return (st - center) * s + center;
}

float scale(in float st, in float s) {
  #ifdef CENTER_2D
  return scale(st,  s, CENTER);
  #else
  return scale(st,  s, .5);
  #endif
}


float2 scale(in float2 st, in float2 s, in float2 center) {
  return (st - center) * s + center;
}

float2 scale(in float2 st, in float value, in float2 center) {
  return scale(st, float2(value, value), center);
}

float2 scale(in float2 st, in float2 s) {
  #ifdef CENTER_2D
  return scale(st,  s, CENTER_2D);
  #else
  return scale(st,  s, float2(.5, .5));
  #endif
}

float2 scale(in float2 st, in float value) {
  return scale(st, float2(value, value));
}

float3 scale(in float3 st, in float3 s, in float3 center) {
  return (st - center) * s + center;
}

float3 scale(in float3 st, in float value, in float3 center) {
  return scale(st, float3(value, value, value), center);
}

float3 scale(in float3 st, in float3 s) {
  #ifdef CENTER_3D
  return scale(st,  s, CENTER_3D);
  #else
  return scale(st,  s, float3(.5, .5, .5));
  #endif
}

float3 scale(in float3 st, in float value) {
  return scale(st, float3(value, value, value));
}
#endif
