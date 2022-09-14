#include "../math/const.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: UV to equirect 3D projected vector 
use: <vec3> cart2equirect(<vec2> uv)
license: |
  Copyright (c) 2022 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_CART2EQUIRECT
#define FNC_CART2EQUIRECT
vec3 cart2equirect(vec2 uv) {
    float Phi = PI - uv.y * PI;
    float Theta = uv.x * 2.0 * PI;
    vec3 dir = vec3(cos(Theta), 0.0, sin(Theta));
    dir.y   = cos(Phi);//clamp(cos(Phi), MinCos, 1.0);
    dir.xz *= sqrt(1.0 - dir.y * dir.y);
    return dir;
}
#endif