/*
contributors: Patricio Gonzalez Vivo
description: Converts from xyY to XYZ
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn xyY2xyz(xyY: vec3f) -> vec3f {
    let Y = xyY.z;  // Y is already in 0-100 scale from rgb2xyY
    let f = 1.0/xyY.y;
    let x = Y * xyY.x * f;
    let z = Y * (1.0 - xyY.x - xyY.y) * f;
    // Output XYZ in 0-100 range (Y is already 0-100, x and z inherit the scale)
    return vec3f(x, Y, z);
}

fn xyY2xyz4(color: vec4f) -> vec4f {
    return vec4f(xyY2xyz(color.xyz), color.a);
}
