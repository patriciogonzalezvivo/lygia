/*
contributors: Patricio Gonzalez Vivo
description: Returns a circle-shaped SDF.
use: circleSDF(vec2 st[, vec2 center])
options:
    - CENTER_2D: vec2, defaults to vec2(.5)
    - CIRCLESDF_FNC(POS_UV): function used to calculate the SDF, defaults to GLSL length function, use lengthSq for a different slope
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define CIRCLESDF_FNC(POS_UV) length(POS_UV)

fn circleSDF(v: vec2f) -> f32 {
    v -= CENTER_2D;
    v -= 0.5;
    return CIRCLESDF_FNC(v) * 2.0;
}
