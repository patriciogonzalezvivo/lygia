import lygia::sdf::circleSDF::circleSDF;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns an almond-shaped (vesica piscis) signed distance field (SDF)
use: |
    vesicaSDF(st: vec2f, w: f32) -> f32
    vesicaSDF(st: vec2f) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn vesicaSDF(st: vec2f, w: f32) -> f32 {
    let offset = vec2f(w * 0.5, 0.0);
    return max(circleSDF(st - offset), circleSDF(st + offset));
}

fn vesicaSDF1(st: vec2f) -> f32 {
    return vesicaSDF(st, 0.5);
}
