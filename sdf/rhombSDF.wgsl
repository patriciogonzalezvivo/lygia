#include "triSDF.wgsl"
#include "../space/scale.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a rhomb-shaped sdf
use: rhombSDF(<vec2> st)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rhombSDF(st: vec2f) -> f32 {
    let offset = 1.0;
    offset = CENTER_2D.y * 2.0;
    return max(triSDF(st),
               triSDF(vec2f(st.x, offset-st.y)));

}
