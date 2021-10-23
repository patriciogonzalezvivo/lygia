/*
author: Patricio Gonzalez Vivo
description: transpose matrixes
use: <mat3|mat4> transpose(in <mat3|mat4> m) 
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    

*/

#ifndef FNC_TRANSPOSE
#define FNC_TRANSPOSE

mat3 transpose(in mat3 m) {
    vec3 i0 = m[0];
    vec3 i1 = m[1];
    vec3 i2 = m[2];

    return mat3(
                    vec3(i0.x, i1.x, i2.x),
                    vec3(i0.y, i1.y, i2.y),
                    vec3(i0.z, i1.z, i2.z)
                );
}

mat4 transpose(in mat4 m) {
    vec4 i0 = m[0];
    vec4 i1 = m[1];
    vec4 i2 = m[2];
    vec4 i3 = m[3];

    return mat4(
                    vec4(i0.x, i1.x, i2.x, i3.x),
                    vec4(i0.y, i1.y, i2.y, i3.y),
                    vec4(i0.z, i1.z, i2.z, i3.z),
                    vec4(i0.w, i1.w, i2.w, i3.w)
                );
}

#endif