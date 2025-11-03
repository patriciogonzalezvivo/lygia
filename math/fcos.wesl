import lygia::math::consts::TWO_PI;

/*
contributors: Inigo Quiles
description: A band-limited variant of cos(x) which reduces aliasing at high frequencies. From https://iquilezles.org/articles/bandlimiting/
use: fcos(value: f32) -> f32
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn fcos(x: f32) -> f32 {
    let w = fwidth(x);
    return cos(x) * smoothstep(TWO_PI, 0.0, w);
}
