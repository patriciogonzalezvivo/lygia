/*
contributors: Patricio Gonzalez Vivo
description: Returns a hexagon-shaped SDF
use: hexSDF(<vec2> st)
options:
    - CENTER_2D: vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn hexSDF(st: vec2f) -> f32 {
    st -= CENTER_2D;
    st *= 2.0;
    st = st * 2.0 - 1.0;
    st = abs(st);
    return max(abs(st.y), st.x * .866025 + st.y * .5);
}
