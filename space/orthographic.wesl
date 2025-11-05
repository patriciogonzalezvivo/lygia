/*
contributors: Patricio Gonzalez Vivo
description: create orthographic matrix
use: <mat4> orthographic(<float> left, <float> right, <float> bottom, <float> top, <float> near, <float> far)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn orthographic(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> mat4x4f {
    return mat4x4f(
        vec4f(2.0 / (r - l), 0.0, 0.0, 0.0),
        vec4f(0.0, 2.0 / (t - b), 0.0, 0.0),
        vec4f(0.0, 0.0, -2.0 / (f - n), 0.0),
        vec4f(-(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n), 1.0)
    );
}
