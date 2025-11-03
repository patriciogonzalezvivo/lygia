/*
contributors: Patricio Gonzalez Vivo
description: returns a 2x2 scale matrix
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn scale2d(s: f32) -> mat2x2<f32> { return mat2x2<f32>(vec2f(s, 0.0), vec2f(0.0, s)); }
fn scale2dVec(s: vec2f) -> mat2x2<f32> { return mat2x2<f32>(vec2f(s.x, 0.0), vec2f(0.0, s.y)); }
fn scale2dXY(x: f32, y: f32) -> mat2x2<f32> { return mat2x2<f32>(vec2f(x, 0.0), vec2f(0.0, y)); }
