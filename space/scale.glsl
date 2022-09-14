/*
original_author: Patricio Gonzalez Vivo
description: scale a 2D space variable
use: scale(<vec2> st, <vec2|float> scale_factor [, <vec2> center])
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


vec2 scale(in vec2 st, in vec2 s, in vec2 center) {
  return (st - center) * s + center;
}

vec2 scale(in vec2 st, in float value, in vec2 center) {
  return scale(st, vec2(value), center);
}

vec2 scale(in vec2 st, in vec2 s) {
  #ifdef CENTER_2D
  return scale(st,  s, CENTER_2D);
  #else
  return scale(st,  s, vec2(.5));
  #endif
}

vec2 scale(in vec2 st, in float value) {
  return scale(st, vec2(value));
}

vec3 scale(in vec3 st, in vec3 s, in vec3 center) {
  return (st - center) * s + center;
}

vec3 scale(in vec3 st, in float value, in vec3 center) {
  return scale(st, vec3(value), center);
}

vec3 scale(in vec3 st, in vec3 s) {
  #ifdef CENTER_3D
  return scale(st,  s, CENTER_3D);
  #else
  return scale(st,  s, vec3(.5));
  #endif
}

vec3 scale(in vec3 st, in float value) {
  return scale(st, vec3(value));
}
#endif
