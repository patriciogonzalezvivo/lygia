/*
contributors: Patricio Gonzalez Vivo
description: 'Moves the center from 0.0 to 0.5'
use: <float|vec2|vec3> uncenter(<float|vec2|vec3> st)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn uncenter(v: f32) -> f32 { return v * 0.5 + 0.5; }
fn uncenter2(v: vec2f) -> vec2f { return v * 0.5 + 0.5; }
fn uncenter3(v: vec3f) -> vec3f { return v * 0.5 + 0.5; }
