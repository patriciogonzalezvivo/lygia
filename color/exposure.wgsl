/*
contributors: Patricio Gonzalez Vivo
description: Change the exposure of a color
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn exposure(color : vec3f, amount : f32) -> vec3f {
    return color * pow(2., amount);
}