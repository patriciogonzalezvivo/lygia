#include "../math/const.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a sdf for a regular polygon with V sides.
use: polySDF(<vec2> st, int V)
options:
    - CENTER_2D: vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn polySDF(st: vec2f, V: i32) -> f32 {
    st -= CENTER_2D;
    st *= 2.0;
    st = st * 2.0 - 1.0;
    let a = atan(st.x, st.y) + PI;
    let r = length(st);
    let v = TAU / float(V);
    return cos(floor(.5 + a / v) * v - a ) * r;
}
