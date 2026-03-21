#include "circleSDF.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns an almond-shaped sdf
use: <float> vesicaSDF(<vec2> st, <float> w)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn vesicaSDF2(st: vec2f, w: f32) -> f32 {
    let offset = vec2f(w*0.5,0.);
    return max( circleSDF(st-offset),
                circleSDF(st+offset));
}

fn vesicaSDF2a(st: vec2f) -> f32 {
    return vesicaSDF(st, 0.5);
}
