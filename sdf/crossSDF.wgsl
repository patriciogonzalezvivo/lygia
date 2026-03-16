#include "rectSDF.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a cross-shaped SDF
use: crossSDF(<vec2> st, size s)
options:
    - CENTER_2D: vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn crossSDF(st: vec2f, s: f32) -> f32 {
    let size = vec2f(.25, s);
    return min(rectSDF(st.xy, size.xy),
               rectSDF(st.xy, size.yx));
}
