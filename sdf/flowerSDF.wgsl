/*
contributors: Patricio Gonzalez Vivo
description: Returns a flower shaped SDF
use: flowerSDF(<vec2> st, <int> n_sides)
options:
    - CENTER_2D: vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn flowerSDF(st: vec2f, N: i32) -> f32 {
    st -= CENTER_2D;
    st -= 0.5;
    st *= 4.0;
    let r = length(st) * 2.0;
    let a = atan(st.y, st.x);
    let v = float(N) * 0.5;
    return 1.0 - (abs(cos(a * v)) *  0.5 + 0.5) / r;
}
