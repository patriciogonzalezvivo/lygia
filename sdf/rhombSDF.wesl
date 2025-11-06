@if(CENTER_2D)
import constants::CENTER_2D;
import lygia::sdf::triSDF::triSDF;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a rhomb-shaped (diamond) signed distance field (SDF)

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default center (0.5, 0.5)
use: |
    rhombSDF(st: vec2f) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rhombSDF(st: vec2f) -> f32 {
    var offset = 1.0;
    @if(CENTER_2D)
    offset = constants::CENTER_2D.y * 2.0;

    return max(triSDF(st), triSDF(vec2f(st.x, offset - st.y)));
}
