/*
author: Bjorn Ottosson (@bjornornorn)
description: linear rgb ot OKLab https://bottosson.github.io/posts/oklab/
use: <vec3\vec4> rgb2oklab(<vec3|vec4> srgb)
license: |
    Copyright (c) 2020 Björn Ottosson
    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef FNC_RGB2OKLAB
#define FNC_RGB2OKLAB
                       
const mat3 inv_oklab_A = mat3(  0.2104542553, 1.9779984951, 0.0259040371,
                                0.7936177850, -2.4285922050, 0.7827717662,
                                -0.0040720468, 0.4505937099, -0.8086757660);

const mat3 inv_oklab_B = mat3(  0.4121656120, 0.2118591070, 0.0883097947,
                                0.5362752080, 0.6807189584, 0.2818474174,
                                0.0514575653, 0.1074065790, 0.6302613616);

vec3 rgb2oklab(vec3 linear) {
    vec3 lms = inv_oklab_B * linear;
    return inv_oklab_A * (sign(lms)*pow(abs(lms), vec3(0.3333333333333)));
    
}

vec4 rgb2oklab(vec4 linear) {
    return vec4(rgb2oklab(linear.rgb), linear.a);
}
#endif