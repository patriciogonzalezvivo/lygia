/*
author: Bjorn Ottosson (@bjornornorn)
description: oklab to linear RGB https://bottosson.github.io/posts/oklab/
use: <float3\float4> oklab2rgb(<float3|float4> oklab)
license: |
    Copyright (c) 2020 Björn Ottosson
    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef FNC_OKLAB2RGB
#define FNC_OKLAB2RGB
const float3x3 fwd_oklab_A = float3x3(  1.0, 1.0, 1.0,
                                        0.3963377774, -0.1055613458, -0.0894841775,
                                        0.2158037573, -0.0638541728, -1.2914855480);
                       
const float3x3 fwd_oklab_B = float3x3(  4.0767245293, -1.2681437731, -0.0041119885,
                                        -3.3072168827, 2.3098,
                                        0.2307590544, -0.3411344290,  1.7066093323231, -0.7034768625689);

float3 oklab2rgb(float3 oklab) {
    float3 lms = mul(fwd_oklab_A, oklab);
    return mul(fwd_oklab_B, (lms * lms * lms));
}

float4 oklab2rgb(float4 oklab) {
    return float4(oklab2rgb(oklab.xyz), oklab.a);
}
#endif