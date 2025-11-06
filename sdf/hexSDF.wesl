@if(CENTER_2D)
import constants::CENTER_2D;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a hexagon-shaped signed distance field (SDF)

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default center (0.5, 0.5)
use: |
    hexSDF(st: vec2f) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn hexSDF(st_in: vec2f) -> f32 {
    var st: vec2f;
    @if(CENTER_2D)
    st = (st_in - constants::CENTER_2D) * 2.0;
    @else
    st = st_in * 2.0 - 1.0;

    let st_abs = abs(st);
    return max(abs(st_abs.y), st_abs.x * 0.866025 + st_abs.y * 0.5);
}
