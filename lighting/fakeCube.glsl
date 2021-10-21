#include "../math/powFast.glsl"
#include "material/shininess.glsl"

/*
author: Patricio Gonzalez Vivo
description: creates a fake cube and returns the value giving a normal direction
use: <vec3> fakeCube(<vec3> _normal [, <float> _shininnes])
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_FAKECUBE
#define FNC_FAKECUBE

vec3 fakeCube(vec3 _normal, float _shininnes) {
    vec3 rAbs = abs(_normal);
    return vec3( powFast(max(max(rAbs.x, rAbs.y), rAbs.z) + 0.005, _shininnes) );
}

vec3 fakeCube(vec3 _normal) {
    return fakeCube(_normal, materialShininess() );
}

#endif