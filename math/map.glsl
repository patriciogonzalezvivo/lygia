#include "saturate.glsl"

/*
author: Johan Ismael
description: Map a value between one range to another.
use: map(<float|vec2|vec3|vec4> value, <float|vec2|vec3|vec4> inMin, <float|vec2|vec3|vec4> inMax, <float|vec2|vec3|vec4> outMin, <float|vec2|vec3|vec4> outMax)
license: |
  Copyright (c) 2017 Johan Ismael.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_MAP
#define FNC_MAP

float map( float value, float inMin, float inMax ) {
    return saturate( (value-inMin)/(inMax-inMin));
}

vec2 map( vec2 value, vec2 inMin, vec2 inMax ) {
    return saturate( (value-inMin)/(inMax-inMin));
}

vec3 map( vec3 value, vec3 inMin, vec3 inMax ) {
    return saturate( (value-inMin)/(inMax-inMin));
}

vec4 map( vec4 value, vec4 inMin, vec4 inMax ) {
    return saturate( (value-inMin)/(inMax-inMin));
}

float map(in float value, in float inMin, in float inMax, in float outMin, in float outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec2 map(in vec2 value, in vec2 inMin, in vec2 inMax, in vec2 outMin, in vec2 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec3 map(in vec3 value, in vec3 inMin, in vec3 inMax, in vec3 outMin, in vec3 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

vec4 map(in vec4 value, in vec4 inMin, in vec4 inMax, in vec4 outMin, in vec4 outMax) {
  return outMin + (outMax - outMin) * (value - inMin) / (inMax - inMin);
}

#endif
