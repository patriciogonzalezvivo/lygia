/*
contributors: Patricio Gonzalez Vivo
description: Returns a rectangular SDF
options:
    - CENTER_2D: vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rectSDF_round(p: vec2f , b: vec2f, r: f32) -> f32 {
    let d = abs(p - 0.5) * 4.2 - b + r;
    return min(max(d.x, d.y), 0.0) + length(vec2f( max(d.x, 0.0), max(d.y, 0.0) )) - r;   
}

fn rectSDF(st: vec2f, s: vec2f) -> f32 {
    let uv = st * 2.0 - 1.0;
    return max( abs(uv.x / s.x),
                abs(uv.y / s.y) );
}