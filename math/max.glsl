/*
author: Patricio Gonzalez Vivo
description: extend GLSL Max function to add more arguments
use: 
  - max(<float> A, <float> B, <float> C[, <float> D])
  - max(<vec2|vec3|vec4> A)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_MAX
#define FNC_MAX
float max(in float a, in float b, in float c) {
  return max(a, max(b, c));
}

float max(in float a, in float b, in float c, in float d) {
  return max(max(a, b), max(c, d));
}

float max(const vec2 v) {
    return max(v.x, v.y);
}

float max(const vec3 v) {
    return max(v.x, v.y, v.z);
}

float max(const vec4 v) {
    return max(v.x, v.y, v.z, v.w);
}
#endif
