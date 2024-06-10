/*
contributors: Patricio Gonzalez Vivo
description: Converts a hue value to a RGB vec3 color.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn hue2rgb(hue: f32) -> vec3f {
    let R = abs(hue * 6.0 - 3.0) - 1.0;
    let G = 2.0 - abs(hue * 6.0 - 2.0);
    let B = 2.0 - abs(hue * 6.0 - 4.0);
    return saturate(vec3f(R,G,B));
}
