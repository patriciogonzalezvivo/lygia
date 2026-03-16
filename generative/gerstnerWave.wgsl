#include "../math/const.wgsl"

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

fn gerstnerWave2(_uv: vec2f, _dir: vec2f, _steepness: f32, _wavelength: f32, _time: f32, _tangent: vec3f, _binormal: vec3f) -> vec3f {
    let k = 2.0 * PI / _wavelength;
    let c = sqrt(9.8 / k);
    let d = normalize(_dir);
    let f = k * (dot(d, _uv) - c * _time);
    let a = _steepness / k;

    _tangent += vec3f(
        -d.x * d.x * (_steepness * sin(f)),
        d.x * (_steepness * cos(f)),
        -d.x * d.y * (_steepness * sin(f))
    );
    _binormal += vec3f(
        -d.x * d.y * (_steepness * sin(f)),
        d.y * (_steepness * cos(f)),
        -d.y * d.y * (_steepness * sin(f))
    );
    return vec3f(
        d.x * (a * cos(f)),
        a * sin(f),
        d.y * (a * cos(f))
    );
}

fn gerstnerWave2a(_uv: vec2f, _dir: vec2f, _steepness: f32, _wavelength: f32, _time: f32, _normal: vec3f) -> vec3f {
    let _tangent = vec3f(0.0);
    let _binormal = vec3f(0.0);
    let pos = gerstnerWave (_uv, _dir, _steepness, _wavelength, _time, _tangent, _binormal);
    _normal = normalize(cross(_binormal, _tangent));
    return pos;
}

fn gerstnerWave2b(_uv: vec2f, _dir: vec2f, _steepness: f32, _wavelength: f32, _time: f32) -> vec3f {
    let _tangent = vec3f(0.0);
    let _binormal = vec3f(0.0);
    return gerstnerWave (_uv, _dir, _steepness, _wavelength, _time, _tangent, _binormal);
}
