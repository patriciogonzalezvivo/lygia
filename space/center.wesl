/*
contributors: Patricio Gonzalez Vivo
description: |
    It center the coordinates from 0 to 1 to -1 to 1
    So the center goes from 0.5 to 0.0.
use: <float|vec2|vec3> center(<float|vec2|vec3> st)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn center(x: f32) -> f32 { return x * 2.0 - 1.0; }
fn center2(v: vec2f) -> vec2f { return v * 2.0 - 1.0; }
fn center3(v: vec3f) -> vec3f { return v * 2.0 - 1.0; }
