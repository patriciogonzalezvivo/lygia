/*
contributors: Patricio Gonzalez Vivo
description: Returns a spiral SDF
use: spiralSDF(<vec2> st, <float> turns)
options:
    - CENTER_2D: vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn spiralSDF(st: vec2f, t: f32) -> f32 {
    st -= CENTER_2D;
    st -= 0.5;
    let r = dot(st, st);
    let a = atan(st.y, st.x);
    return abs(sin(fract(log(r) * t + a * 0.159)));
}
