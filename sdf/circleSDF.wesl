@if(CENTER_2D)
import constants::CENTER_2D;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a circle-shaped signed distance field (SDF)

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default center (0.5, 0.5)
use: |
    circleSDF(st: vec2f) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn circleSDF(st: vec2f) -> f32 {
    var v: vec2f;
    @if(CENTER_2D)
    v = st - constants::CENTER_2D;
    @else
    v = st - 0.5;

    return length(v) * 2.0;
}
