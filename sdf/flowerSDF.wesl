@if(CENTER_2D)
import constants::CENTER_2D;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a flower-shaped signed distance field (SDF)

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default center (0.5, 0.5)
use: |
    flowerSDF(st: vec2f, N: i32) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn flowerSDF(st_in: vec2f, N: i32) -> f32 {
    var st: vec2f;
    @if(CENTER_2D)
    st = st_in - constants::CENTER_2D;
    @else
    st = st_in - 0.5;

    st *= 4.0;
    let r = length(st) * 2.0;
    // Note: flowerSDF is rotationally symmetric, so Y-down doesn't affect appearance
    let a = atan2(st.y, st.x);
    let v = f32(N) * 0.5;
    return 1.0 - (abs(cos(a * v)) * 0.5 + 0.5) / r;
}
