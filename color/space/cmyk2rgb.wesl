/*
contributors: Patricio Gonzalez Vivo
description: Convert CMYK to RGB
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn cmyk2rgb(cmyk: vec4f) -> vec3f {
    let invK: f32 = 1.0 - cmyk.w;
    return saturate(1.0 - min(vec3f(1.0), cmyk.xyz * invK + cmyk.w));
}
