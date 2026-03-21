#include "../math/const.wgsl"
#include "../space/scale.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a star-shaped sdf with V branches
use: starSDF(<vec2> st, <int> V, <float> scale)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn starSDF2(st: vec2f, V: i32, s: f32) -> f32 {
    st -= CENTER_2D;
    st -= 0.5;
    st *= 2.0;
    let a = atan(st.y, st.x) / TAU;
    let seg = a * float(V);
    a = ((floor(seg) + 0.5) / float(V) +
        mix(s, -s, step(0.5, fract(seg))))
        * TAU;
    return abs(dot(vec2f(cos(a), sin(a)),
                   st));
}

fn starSDF2a(st: vec2f, V: i32) -> f32 {
    return starSDF( scale(st, 12.0/float(V)), V, 0.1);
}
