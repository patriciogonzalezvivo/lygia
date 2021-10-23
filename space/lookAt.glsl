/*
author: Patricio Gonzalez Vivo
description: create a look up matrix 
use: 
    - <mat3> lookAt(<vec3> forward, <vec3> up)
    - <mat3> lookAt(<vec3> target, <vec3> eye, <vec3> up)
    - <mat3> lookAt(<vec3> target, <vec3> eye, <float> rolle)

license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_LOOKAT
#define FNC_LOOKAT

mat3 lookAt(vec3 forward, vec3 up) {
    vec3 xaxis = normalize(cross(forward, up));
    vec3 yaxis = up;
    vec3 zaxis = forward;
    return mat3(xaxis, yaxis, zaxis);
}

mat3 lookAt(vec3 target, vec3 eye, vec3 up) {
    vec3 zaxis = normalize(target - eye);
    vec3 xaxis = normalize(cross(zaxis, up));
    vec3 yaxis = cross(zaxis, xaxis);
    return mat3(xaxis, yaxis, zaxis);
}

mat3 lookAt(vec3 target, vec3 eye, float roll) {
    vec3 up = vec3(sin(roll), cos(roll), 0.0);
    vec3 zaxis = normalize(target - eye);
    vec3 xaxis = normalize(cross(zaxis, up));
    vec3 yaxis = normalize(cross(xaxis, zaxis));
    return mat3(xaxis, yaxis, zaxis);
}

#endif