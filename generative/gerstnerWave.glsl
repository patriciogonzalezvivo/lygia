#include "../math/const.glsl"

/*
author: Patricio Gonzalez Vivo
description: Gerstner Wave generator based on this tutorial https://catlikecoding.com/unity/tutorials/flow/waves/
use: 
    - <vec3> gerstnerWave (<vec2> uv, <vec2> dir, <float> steepness, <float> wavelength, <float> _time [, inout <vec3> _tangent, inout <vec3> _binormal] )
    - <vec3> gerstnerWave (<vec2> uv, <vec2> dir, <float> steepness, <float> wavelength, <float> _time [, inout <vec3> _normal] )
license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FNC_GERSTNERWAVE
#define FNC_GERSTNERWAVE

vec3 gerstnerWave (in vec2 _uv, in vec2 _dir, in float _steepness, in float _wavelength, in float _time, inout vec3 _tangent, inout vec3 _binormal) {
    float k = 2.0 * PI / _wavelength;
    float c = sqrt(9.8 / k);
    vec2 d = normalize(_dir);
    float f = k * (dot(d, _uv) - c * _time);
    float a = _steepness / k;

    _tangent += vec3(
        -d.x * d.x * (_steepness * sin(f)),
        d.x * (_steepness * cos(f)),
        -d.x * d.y * (_steepness * sin(f))
    );
    _binormal += vec3(
        -d.x * d.y * (_steepness * sin(f)),
        d.y * (_steepness * cos(f)),
        -d.y * d.y * (_steepness * sin(f))
    );
    return vec3(
        d.x * (a * cos(f)),
        a * sin(f),
        d.y * (a * cos(f))
    );
}

vec3 gerstnerWave (in vec2 _uv, in vec2 _dir, in float _steepness, in float _wavelength, in float _time, inout vec3 _normal) {
    vec3 _tangent = vec3(0.0);
    vec3 _binormal = vec3(0.0);
    vec3 pos = gerstnerWave (_uv, _dir, _steepness, _wavelength, _time, _tangent, _binormal);
    _normal = normalize(cross(_binormal, _tangent));
    return pos;
}

vec3 gerstnerWave (in vec2 _uv, in vec2 _dir, in float _steepness, in float _wavelength, in float _time) {
    vec3 _tangent = vec3(0.0);
    vec3 _binormal = vec3(0.0);
    return gerstnerWave (_uv, _dir, _steepness, _wavelength, _time, _tangent, _binormal);
}

#endif