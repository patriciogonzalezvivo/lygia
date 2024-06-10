/*
contributors: Patricio Gonzalez Vivo
description: sampling function to make a texture behave like GL_NEAREST
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn nearest(v: vec2f, res: vec2f) -> vec2f {
    let offset = 0.5 / (res - 1.0);
    return floor(v * res) / res + offset;
}