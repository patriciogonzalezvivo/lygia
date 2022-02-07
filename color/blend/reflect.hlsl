/*
author: Jamie Owen
description: Photoshop Reflect blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendReflect(<float|float3> base, <float|float3> blend [, <float> opacity])
licence:  |
    The MIT License (MIT) Copyright (c) 2015 Jamie Owen
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_BLENDREFLECT
#define FNC_BLENDREFLECT
float blendReflect(in float base, in float blend) {
    return (blend == 1.)? blend : min(base * base / (1. - blend), 1.);
}

float3 blendReflect(in float3 base, in float3 blend) {
    return float3(  blendReflect(base.r, blend.r),
                    blendReflect(base.g, blend.g),
                    blendReflect(base.b, blend.b) );
}

float3 blendReflect(in float3 base, in float3 blend, in float opacity) {
    return (blendReflect(base, blend) * opacity + base * (1. - opacity));
}
#endif
