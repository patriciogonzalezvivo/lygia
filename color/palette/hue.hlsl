/*
author: Patricio Gonzalez Vivo
description: Physical Hue. Ratio: 1/3 = neon, 1/4 = refracted, 1/5+ = approximate white
use: <float3> hue(<float> hue[, <float> ratio])
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_PALETTE_HUE
#define FNC_PALETTE_HUE

float3 hue(float hue, float ratio) {
    return smoothstep(  float3(0.9059, 0.8745, 0.8745),
                        float3(1.0), 
                        abs( mod(hue + float3(0.0,1.0,2.0) * ratio, 1.0) * 2.0 - 1.0));
}

float3 hue(float hue) { return hue(hue, 0.33333); }

#endif