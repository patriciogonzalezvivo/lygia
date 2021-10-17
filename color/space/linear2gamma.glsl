/*
author: Patricio Gonzalez Vivo
description: convert from linear to gamma color space.
use: linear2gamma(<float|vec3|vec4> color)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#if !defined(GAMMA) && !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
#define GAMMA 2.2
#endif

#ifndef FNC_LINEAR2GAMMA
#define FNC_LINEAR2GAMMA
vec3 linear2gamma(in vec3 v) {
#ifdef GAMMA
    return pow(v, vec3(1. / GAMMA));
#else
    // assume gamma 2.0
    return sqrt(v);
#endif
}

vec4 linear2gamma(in vec4 v) {
    return vec4(linear2gamma(v.rgb), v.a);
}

float linear2gamma(in float v) {
#ifdef GAMMA
    return pow(v, 1. / GAMMA);
#else
    // assume gamma 2.0
    return sqrt(v);
#endif
}
#endif
