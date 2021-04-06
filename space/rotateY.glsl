#include "../math/rotate4dY.glsl"

/*
author: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian angle
use: rotateY(<vec3> pos, float radian [, vec4 center])
options:
  - CENTER_3D
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_ROTATEY
#define FNC_ROTATEY
vec3 rotateY(in vec3 pos, in float radian, in vec3 center) {
  return (rotate4dY(radian) * vec4((pos - center), 1.)).xyz + center;
}

vec3 rotateY(in vec3 pos, in float radian) {
  #ifdef CENTER_3D
  return rotateY(pos, radian, CENTER_3D);
  #else
  return rotateY(pos, radian, vec3(.0));
  #endif
}
#endif
