/*
contributors: Patricio Gonzalez Vivo
description: scale a 2D space variable
options:
    - CENTER_2D
    - CENTER_3D
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn scale(st: vec2f, s: vec2f) -> vec2f {
    return (st - 0.5) * s + 0.5;
}
