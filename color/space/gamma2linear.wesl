/*
contributors: Patricio Gonzalez Vivo
description: Convert from gamma to linear color space.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn gamma2linear(gamma: vec3f) -> vec3f {
    return pow(gamma, vec3(2.2));
}
