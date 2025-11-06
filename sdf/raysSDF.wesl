@if(CENTER_2D)
import constants::CENTER_2D;
import lygia::math::consts::TAU;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a signed distance field (SDF) for rays with N branches

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default center (0.5, 0.5)
use: |
    raysSDF(st: vec2f, N: i32) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn raysSDF(st_in: vec2f, N: i32) -> f32 {
    var st: vec2f;
    @if(CENTER_2D)
    st = st_in - constants::CENTER_2D;
    @else
    st = st_in - 0.5;

    // Negate Y to compensate for WGSL's Y-down coordinates (maintains GLSL ray pattern)
    st.y = -st.y;

    return fract(atan2(st.y, st.x) / TAU * f32(N));
}
