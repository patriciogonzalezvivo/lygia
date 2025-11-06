@if(CENTER_2D)
import constants::CENTER_2D;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a triangle-shaped signed distance field (SDF)

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default center (0.5, 0.5)
use: |
    triSDF(st: vec2f) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn triSDF(st_in: vec2f) -> f32 {
    var st: vec2f;
    @if(CENTER_2D)
    st = (st_in - constants::CENTER_2D) * 5.0;
    @else
    st = (st_in - 0.5) * 5.0;

    // Flip y-axis to compensate for WebGPU's y-down coordinate system
    st.y = -st.y;

    return max(abs(st.x) * 0.866025 + st.y * 0.5, -st.y * 0.5);
}
