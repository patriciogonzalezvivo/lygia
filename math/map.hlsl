/*
author: Johan Ismael
description: Map a value between one range to another.
use: map(<float|float2|float3|float4> value, <float|float2|float3|float4> inMin, <float|float2|float3|float4> inMax, <float|float2|float3|float4> outMin, <float|float2|float3|float4> outMax)
license: |
  Copyright (c) 2017 Johan Ismael.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_MAP
#define FNC_MAP

float map(in float value, in float inMin, in float inMax, in float outMin, in float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float2 map(in float2 value, in float2 inMin, in float2 inMax, in float2 outMin, in float2 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float3 map(in float3 value, in float3 inMin, in float3 inMax, in float3 outMin, in float3 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

float4 map(in float4 value, in float4 inMin, in float4 inMax, in float4 outMin, in float4 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

#endif
