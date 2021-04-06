/*
author: Patricio Gonzalez Vivo
description: |
  Convert from RGB to YIQ which was the followin range
  Y [0,.1], I [-0.5957, 0.5957], Q [-0.5226, 0.5226]
  From https://en.wikipedia.org/wiki/YIQ
use: rgb2yiq(<vec3|vec4> color)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_RGB2YIQ
#define FNC_RGB2YIQ
// https://en.wikipedia.org/wiki/YIQ
const mat3 rgb2yiq_mat = mat3(
  .299,  .596,  .211,
  .587, -.274, -.523,
  .114, -.322,  .0312
);

vec3 rgb2yiq(in vec3 rgb) {
  return rgb2yiq_mat * rgb;
}

vec4 rgb2yiq(in vec4 rgb) {
    return vec4(rgb2yiq(rgb.rgb), rgb.a);
}
#endif
