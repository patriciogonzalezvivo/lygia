#include "../math/const.glsl"

/*
author: Patricio Gonzalez Vivo
description: UV to equirect 3D fisheye vector 
use: <vec3> cart2equirect(<vec2> uv, <float> MinCos)
license: |
  Copyright (c) 2022 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_CART2FISHEYE
#define FNC_CART2FISHEYE
vec3 cart2fisheye(vec2 uv) {
    vec2 ndc = uv * 2.0 - 1.0;
    ndc.x *= u_resolution.x / u_resolution.y;
    float R = sqrt(ndc.x * ndc.x + ndc.y * ndc.y);
    vec3 dir = vec3(ndc.x / R, 0.0, ndc.y / R);
    float Phi = (R) * PI * 0.52;
    dir.y   = cos(Phi);//clamp(, MinCos, 1.0);
    dir.xz *= sqrt(1.0 - dir.y * dir.y);
    return dir;
}
#endif