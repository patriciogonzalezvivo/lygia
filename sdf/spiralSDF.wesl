@if(CENTER_2D)
import constants::CENTER_2D;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a spiral-shaped signed distance field (SDF)

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default center (0.5, 0.5)
use: |
    spiralSDF(st: vec2f, t: f32) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn spiralSDF(st_in: vec2f, t: f32) -> f32 {
    var st: vec2f;
    @if(CENTER_2D)
    st = st_in - constants::CENTER_2D;
    @else
    st = st_in - 0.5;

    // Negate Y to compensate for WGSL's Y-down coordinates (maintains GLSL spiral direction)
    st.y = -st.y;

    let r = dot(st, st);
    let a = atan2(st.y, st.x);
    return abs(sin(fract(log(r) * t + a * 0.159)));
}
