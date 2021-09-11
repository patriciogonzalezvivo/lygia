/*
author: Patricio Gonzalez Vivo
description: An implementation of mod that matches the GLSL mod.
  Note that HLSL's fmod is different.
use: mod(<float|float2|float3|float4> value, <float|float2|float3|float4> modulus)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_MOD
#define FNC_MOD
float2 mod(float x, float y) {
    return x - y * floor(x / y);
}

float2 mod(float2 x, float2 y) {
    return x - y * floor(x / y);
}

float3 mod(float3 x, float3 y) {
    return x - y * floor(x / y);
}

float4 mod(float4 x, float4 y) {
    return x - y * floor(x / y);
}
#endif