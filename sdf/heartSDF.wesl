@if(CENTER_2D)
import constants::CENTER_2D;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a heart-shaped signed distance field (SDF)

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default center (0.5, 0.5)
use: |
    heartSDF(st: vec2f) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn heartSDF(st_in: vec2f) -> f32 {
    var st: vec2f;
    @if(CENTER_2D)
    st = st_in - constants::CENTER_2D;
    @else
    st = st_in - 0.5;

    // Flip y-axis to compensate for WebGPU's y-down coordinate system
    st.y = -st.y;

    st -= vec2f(0.0, 0.3);
    let r = length(st) * 5.0;
    let st_norm = normalize(st);
    return r - ((st_norm.y * pow(abs(st_norm.x), 0.67)) / (st_norm.y + 1.5) - 2.0 * st_norm.y + 1.26);
}
