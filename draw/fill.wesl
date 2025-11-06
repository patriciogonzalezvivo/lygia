/*
contributors: Patricio Gonzalez Vivo
description: |
    Fill a signed distance field (SDF). From PixelSpiritDeck https://github.com/patriciogonzalezvivo/PixelSpiritDeck
use: |
    fill(sdf: f32, size: f32) -> f32
    fillEdge(sdf: f32, size: f32, edge: f32) -> f32
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

import lygia::math::aastep::aastep;

fn fill(x: f32, size: f32) -> f32 {
    return 1.0 - aastep(size, x);
}

fn fillEdge(x: f32, size: f32, edge: f32) -> f32 {
    return 1.0 - smoothstep(size - edge, size + edge, x);
}
