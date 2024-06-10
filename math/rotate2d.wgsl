/*
contributors: Patricio Gonzalez Vivo
description: returns a 2x2 rotation matrix
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rotate2d(radians: f32) -> mat2x2<f32> {
    let c = cos(radians);
    let s = sin(radians);
    return mat2x2<f32>(c, -s, s, c);
}
