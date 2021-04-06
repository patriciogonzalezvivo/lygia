#include "../math/rotate2d.glsl"
#include "../math/rotate4d.glsl"

/*
author: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian radians
use: rotate(<vec3|vec2> st, float radians [, vec2 center])
options:
  - CENTER_2D
  - CENTER_3D
  - CENTER_4D
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_ROTATE
#define FNC_ROTATE
vec2 rotate(in vec2 st, in float radians, in vec2 center) {
  return rotate2d(radians) * (st - center) + center;
}

vec2 rotate(in vec2 st, in float radians) {
  #ifdef CENTER_2D
  return rotate(st, radians, CENTER_2D);
  #else
  return rotate(st, radians, vec2(.5));
  #endif
}

vec3 rotate(in vec3 xyz, in float radians, in vec3 axis, in vec3 center) {
  return (rotate4d(axis, radians) * vec4(xyz - center, 1.)).xyz + center;
}

vec3 rotate(in vec3 xyz, in float radians, in vec3 axis) {
  #ifdef CENTER_3D
  return rotate(xyz, radians, axis, CENTER_3D);
  #else
  return rotate(xyz, radians, axis, vec3(0.));
  #endif
}

vec4 rotate(in vec4 xyzw, in float radians, in vec3 axis, in vec4 center) {
  return rotate4d(axis, radians) * (xyzw - center) + center;
}

vec4 rotate(in vec4 xyzw, in float radians, in vec3 axis) {
  #ifdef CENTER_4D
  return rotate(xyzw, radians, axis, CENTER_4D);
  #else
  return rotate(xyzw, radians, axis, vec4(0.));
  #endif
}
#endif
