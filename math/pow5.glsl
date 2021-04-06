/*
author: Patricio Gonzalez Vivo
description: power of 5
use: pow5(<float|vec2|vec3|vec4> x)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_POW5
#define FNC_POW5
float pow5(in float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

vec2 pow5(in vec2 x) {
    vec2 x2 = x * x;
    return x2 * x2 * x;
}

vec3 pow5(in vec3 x) {
    vec3 x2 = x * x;
    return x2 * x2 * x;
}

vec4 pow5(in vec4 x) {
    vec4 x2 = x * x;
    return x2 * x2 * x;
}
#endif
