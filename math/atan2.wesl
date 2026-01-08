import lygia::math::consts::{PI, TAU};

/*
contributors: Alexander Griffis
description: |
    The range here in degrees is 0 to pi (0-180 degrees) and -pi to 0 (181 to 359 degrees
    More about it at https://github.com/Yaazarai/GLSL-ATAN2-DOC

use: atan2Custom(y: f32, x: f32) -> f32
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn atan2Custom(y: f32, x: f32) -> f32 {
    return (atan2(y, x) + PI) % TAU;
}
