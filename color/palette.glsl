#include "../math/const.glsl"

/*
author: Inigo Quiles
description:  Procedural generation of color palette algorithm explained here http://www.iquilezles.org/www/articles/palettes/palettes.htm)
use: palette(<float> t, <vec3|vec4> a, <vec3|vec4> b, <vec3|vec4> c, <vec3|vec4> d)
license: |
  Copyright © 2015 Inigo Quilez
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_PALETTE
#define FNC_PALETTE
vec3 palette (in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b * cos(TAU * ( c * t + d ));
}

vec4 palette (in float t, in vec4 a, in vec4 b, in vec4 c, in vec4 d) {
    return a + b * cos(TAU * ( c * t + d ));
}
#endif
