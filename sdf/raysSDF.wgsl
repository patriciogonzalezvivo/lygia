#include "../math/const.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a sdf for rays with N branches
use: raysSDF(<vec2> st, <int> N)
options:
    - CENTER_2D: vec2, defaults to vec2(.5)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn raysSDF(st: vec2f, N: i32) -> f32 {
    st -= CENTER_2D;
    st -= 0.5;
    return fract(atan(st.y, st.x) / TAU * float(N));
}
