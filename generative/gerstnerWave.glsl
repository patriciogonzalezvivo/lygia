#include "../math/const.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Gerstner Wave generator based on this tutorial https://catlikecoding.com/unity/tutorials/flow/waves/
use:
    - <vec3> gerstnerWave (<vec2> uv, <vec2> dir, <float> steepness, <float> wavelength, <float> _time [, inout <vec3> _tangent, inout <vec3> _binormal] )
    - <vec3> gerstnerWave (<vec2> uv, <vec2> dir, <float> steepness, <float> wavelength, <float> _time [, inout <vec3> _normal] )
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
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