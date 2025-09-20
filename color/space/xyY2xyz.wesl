/*
contributors: Patricio Gonzalez Vivo
description: Converts from xyY to XYZ
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn xyY2xyz(xyY: vec3f) -> vec3f {
    let Y = xyY.z;
    let f = 1.0/xyY.y;
    let x = Y * xyY.x * f;
    let z = Y * (1.0 - xyY.x - xyY.y) * f;
    return vec3f(x, Y, z);
}
