@if(CENTER_2D)
import constants::CENTER_2D;
import lygia::math::consts::TAU;
import lygia::space::scale::scale2_f;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a star-shaped signed distance field (SDF) with V branches

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default center (0.5, 0.5)
use: |
    starSDF(st: vec2f, V: i32, s: f32) -> f32
    starSDF(st: vec2f, V: i32) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn starSDF(st_in: vec2f, V: i32, s: f32) -> f32 {
    var st: vec2f;
    @if(CENTER_2D)
    st = st_in - constants::CENTER_2D;
    @else
    st = st_in - 0.5;

    st *= 2.0;
    // Note: starSDF is rotationally symmetric, so Y-down doesn't affect appearance
    let a = atan2(st.y, st.x) / TAU;
    let seg = a * f32(V);
    let aa = ((floor(seg) + 0.5) / f32(V) +
              mix(s, -s, step(0.5, fract(seg)))) * TAU;
    return abs(dot(vec2f(cos(aa), sin(aa)), st));
}

fn starSDF1(st: vec2f, V: i32) -> f32 {
    return starSDF(scale2_f(st, 12.0 / f32(V)), V, 0.1);
}
