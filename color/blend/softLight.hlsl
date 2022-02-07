/*
author: Jamie Owen
description: Photoshop Soft Light blend mode mplementations sourced from this article on https://mouaif.wordpress.com/2009/01/05/photoshop-math-with-glsl-shaders/
use: blendSoftLight(<float|float3|float4> base, <float|float3|float4> blend [, <float> opacity])
licence:  |
    The MIT License (MIT) Copyright (c) 2015 Jamie Owen
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_BLENDSOFTLIGHT
#define FNC_BLENDSOFTLIGHT
float blendSoftLight(in float base, in float blend) {
    return (blend < .5)? (2. * base * blend + base * base * (1. - 2.*blend)): (sqrt(base) * (2. * blend - 1.) + 2. * base * (1. - blend));
}

float3 blendSoftLight(in float3 base, in float3 blend) {
    return float3(  blendSoftLight(base.r, blend.r),
                    blendSoftLight(base.g, blend.g),
                    blendSoftLight(base.b, blend.b) );
}

float4 blendSoftLight(in float4 base, in float4 blend) {
    return float4(  blendSoftLight( base.r, blend.r ),
                    blendSoftLight( base.g, blend.g ),
                    blendSoftLight( base.b, blend.b ),
                    blendSoftLight( base.a, blend.a )
    );
}

float3 blendSoftLight(in float3 base, in float3 blend, in float opacity) {
    return (blendSoftLight(base, blend) * opacity + base * (1. - opacity));
}
#endif
