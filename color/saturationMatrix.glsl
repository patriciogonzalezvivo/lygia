/*
author: Patricio Gonzalez Vivo
description: generate a matrix to change a the saturation of any color
use: saturationMatrix(<float> amount)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_SATURATIONMATRIX
#define FNC_SATURATIONMATRIX
mat4 saturationMatrix(in float amount) {
    vec3 lum = vec3(.3086, .6094, .0820 );

    float invAmount= 1. - amount;

    vec3 red = vec3(lum.x * invAmount);
    red += vec3(amount, .0, .0);

    vec3 green = vec3(lum.y * invAmount);
    green += vec3( .0, amount, .0);

    vec3 blue = vec3(lum.z * invAmount);
    blue += vec3( .0, .0, amount);

    return mat4(red,        .0,
                green,      .0,
                blue,       .0,
                .0, .0, .0, 1.);
}
#endif
