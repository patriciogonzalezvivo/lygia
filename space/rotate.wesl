import package::math::rotate2d::rotate2d;

/*
contributors: Patricio Gonzalez Vivo
description: rotate a 2D space by a radian r
options:
    - CENTER_2D
    - CENTER_3D
    - CENTER_4D
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rotate(st: vec2f, radians: f32) -> vec2f {
    return rotate2d(radians) * (st - 0.5) + 0.5;
}
