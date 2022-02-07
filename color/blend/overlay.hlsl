/*
author: Jamie Owen
description: Photoshop Overlay blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendOverlay(<float|float3> base, <float|float3> blend [, <float> opacity])
licence:  |
    The MIT License (MIT) Copyright (c) 2015 Jamie Owen
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_BLENDOVERLAY
#define FNC_BLENDOVERLAY
float blendOverlay(in float base, in float blend) {
    return (base < .5)? (2.*base*blend): (1. - 2. * (1. - base) * (1. - blend));
}

float3 blendOverlay(in float3 base, in float3 blend) {
    return float3(  blendOverlay(base.r, blend.r),
                    blendOverlay(base.g, blend.g),
                    blendOverlay(base.b, blend.b) );
}

float3 blendOverlay(in float3 base, in float3 blend, in float opacity) {
    return (blendOverlay(base, blend) * opacity + base * (1. - opacity));
}
#endif
