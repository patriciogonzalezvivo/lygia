#include "../math/const.hlsl"

/*
author: Inigo Quiles
description:  Procedural generation of color palette algorithm explained here http://www.iquilezles.org/www/articles/palettes/palettes.htm)
use: palette(<float> t, <float3|float4> a, <float3|float4> b, <float3|float4> c, <float3|float4> d)
license: |
  Copyright © 2015 Inigo Quilez
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef FNC_PALETTE
#define FNC_PALETTE
float3 palette (in float t, in float3 a, in float3 b, in float3 c, in float3 d) {
    return a + b * cos(TAU * ( c * t + d ));
}

float4 palette (in float t, in float4 a, in float4 b, in float4 c, in float4 d) {
    return a + b * cos(TAU * ( c * t + d ));
}
#endif
