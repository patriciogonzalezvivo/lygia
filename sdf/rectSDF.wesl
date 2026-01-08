@if(CENTER_2D)
import constants::CENTER_2D;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Returns a rectangular signed distance field (SDF)

    WESL conditionals:
    - @if(CENTER_2D): Uses custom 2D center point instead of default centering (st * 2.0 - 1.0)
use: |
    rectSDF(st: vec2f, s: vec2f) -> f32
    rectSDF1(st: vec2f, s: f32) -> f32
    rectSDFDefault(st: vec2f) -> f32
    rectSDF3(p: vec2f, b: vec2f, r: f32) -> f32
    rectSDF2Round(p: vec2f, b: f32, r: f32) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rectSDF3(p: vec2f, b: vec2f, r: f32) -> f32 {
    let d = abs(p - 0.5) * 4.2 - b + vec2f(r);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0))) - r;
}

fn rectSDF2Round(p: vec2f, b: f32, r: f32) -> f32 {
    return rectSDF3(p, vec2f(b), r);
}

fn rectSDF(st: vec2f, s: vec2f) -> f32 {
    var uv: vec2f;
    @if(CENTER_2D)
    uv = (st - constants::CENTER_2D) * 2.0;
    @else
    uv = st * 2.0 - 1.0;

    return max( abs(uv.x / s.x),
                abs(uv.y / s.y) );
}

fn rectSDF1(st: vec2f, s: f32) -> f32 {
    return rectSDF(st, vec2f(s));
}

fn rectSDFDefault(st: vec2f) -> f32 {
    return rectSDF(st, vec2f(1.0));
}
