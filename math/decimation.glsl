/*
author: Patricio Gonzalez Vivo
description: decimate a value with an specific presicion 
use: decimation(<float|vec2|vec3|vec4> value, <float|vec2|vec3|vec4> presicion)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_DECIMATION
#define FNC_DECIMATION
float decimation(float value, float presicion) {
    return floor(value * presicion)/presicion;
}

vec2 decimation(vec2 value, float presicion) {
    return floor(value * presicion)/presicion;
}

vec3 decimation(vec3 value, float presicion) {
    return floor(value * presicion)/presicion;
}

vec4 decimation(vec4 value, float presicion) {
    return floor(value * presicion)/presicion;
}

vec2 decimation(vec2 value, vec2 presicion) {
    return floor(value * presicion)/presicion;
}

vec3 decimation(vec3 value, vec3 presicion) {
    return floor(value * presicion)/presicion;
}

vec4 decimation(vec4 value, vec4 presicion) {
    return floor(value * presicion)/presicion;
}
#endif