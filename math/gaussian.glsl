/*
author: Patricio Gonzalez Vivo
description: gaussian coeficient
use: <vec4|vec3|vec2|float> gaussian(<float> sigma, <vec4|vec3|vec2|float> d)
license: |
    Copyright (c) 2022 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_GAUSSIAN
#define FNC_GAUSSIAN

float gaussian(float sigma, float d) {
    return exp(-(d*d) / (2.0 * sigma*sigma));
}

float gaussian(float sigma, vec2 d) {
    return exp(-( d.x*d.x + d.y*d.y) / (2.0 * sigma*sigma));
}

float gaussian(float sigma, vec3 d) {
    return exp(-( d.x*d.x + d.y*d.y + d.z*d.z ) / (2.0 * sigma*sigma));
}

float gaussian(float sigma, vec4 d) {
    return exp(-( d.x*d.x + d.y*d.y + d.z*d.z + d.w*d.w ) / (2.0 * sigma*sigma));
}

#endif