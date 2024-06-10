/*
contributors: Patricio Gonzalez Vivo
description: Convert from linear to gamma color space.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn linear2gamma(v : vec3f) -> vec3f {
    return pow(v, vec3f(1. / 2.2));
}
