/*
contributors: Patricio Gonzalez Vivo
description: Converts from XYZ to xyY space (Y is the luminance)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn xyz2xyY(xyz: vec3f) -> vec3f {
    let Y = xyz.y;
    let f = 1.0 / (xyz.x + xyz.y + xyz.z);
    let x = xyz.x * f;
    let y = xyz.y * f;
    return vec3f(x, y, Y);
}