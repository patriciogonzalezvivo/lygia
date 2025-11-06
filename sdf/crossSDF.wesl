import lygia::sdf::rectSDF::rectSDF;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a cross-shaped signed distance field (SDF)
use: |
    crossSDF(st: vec2f, s: f32) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn crossSDF(st: vec2f, s: f32) -> f32 {
    let size = vec2f(0.25, s);
    return min(rectSDF(st, size), rectSDF(st, size.yx));
}
