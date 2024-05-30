/*
contributors: Patricio Gonzalez Vivo
description: returns a 2x2 scale matrix
*/

fn scale2d(s: vec2f) -> mat2x2<f32> { return mat2x2<f32>(s.x, 0.0, 0.0, s.y); }
