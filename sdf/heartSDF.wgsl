/*
contributors: Patricio Gonzalez Vivo
description: Returns a heart shaped SDF
use: heartSDF(<vec2> st)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn heartSDF(st: vec2f) -> f32 {
    st -= CENTER_2D;
    st -= 0.5;
    st -= vec2f(0.0, 0.3);
    let r = length(st) * 5.0;
    st = normalize(st);
    return r - ((st.y * pow(abs(st.x), 0.67)) / (st.y + 1.5) - (2.0) * st.y + 1.26);
}
