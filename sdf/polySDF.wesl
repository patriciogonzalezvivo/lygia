@if(CENTER_2D)
import constants::CENTER_2D;
import lygia::math::consts::{PI, TAU};

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a signed distance field (SDF) for a regular polygon with V sides

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default center (0.5, 0.5)
use: |
    polySDF(st: vec2f, V: i32) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn polySDF(st_in: vec2f, V: i32) -> f32 {
    var st: vec2f;
    @if(CENTER_2D)
    st = (st_in - constants::CENTER_2D) * 2.0;
    @else
    st = st_in * 2.0 - 1.0;

    // Negate Y to compensate for WGSL's Y-down coordinates (maintains GLSL polygon orientation)
    st.y = -st.y;

    let a = atan2(st.x, st.y) + PI;
    let r = length(st);
    let v = TAU / f32(V);
    return cos(floor(0.5 + a / v) * v - a) * r;
}
