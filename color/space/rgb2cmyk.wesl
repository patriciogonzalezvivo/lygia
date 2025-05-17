/*
contributors: Patricio Gonzalez Vivo
description: Convert CMYK to RGB
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rgb2cmyk(rgb: vec3f) -> vec4f {
    let k = min(1.0 - rgb.r, min(1.0 - rgb.g, 1.0 - rgb.b));
    let invK = 1.0 - k;
    var cmy = (1.0 - rgb - k) / invK;
    cmy *= step(0.0, invK);
    return saturate(vec4f(cmy, k));
}