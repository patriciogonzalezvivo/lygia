/*
contributors: Patricio Gonzalez Vivo
description: Generate a matrix to change a the contrast of any color
use: contrastMatrix(<f32> amount)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_brightnessContrastMatrix.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn contrastMatrix(a: f32) -> mat4x4f {
    let t = (1.0 - a) * 0.5;
    return mat4x4f(
        vec4f(a, 0.0, 0.0, 0.0),
        vec4f(0.0, a, 0.0, 0.0),
        vec4f(0.0, 0.0, a, 0.0),
        vec4f(t, t, t, 1.0)
    );
}
