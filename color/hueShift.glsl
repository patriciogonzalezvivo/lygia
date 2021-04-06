#include "space/rgb2hsv.glsl"
#include "space/hsv2rgb.glsl"

/*
author: Johan Ismael
description: shifts color hue
use: hueShift(<vec3|vec4> color, <float> amount)
license: |
  Copyright (c) 2017 Johan Ismael.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_HUESHIFT
#define FNC_HUESHIFT
vec3 hueShift(in vec3 color, in float amount) {
    vec3 hsv = rgb2hsv(color);
    hsv.r += amount;
    return hsv2rgb(hsv);
}

vec4 hueShift(in vec4 color, in float amount) {
    return vec4(hueShift(color.rgb, amount), color.a);
}
#endif