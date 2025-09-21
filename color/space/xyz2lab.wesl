/*
contributors: Patricio Gonzalez Vivo
description: Converts a XYZ color to Lab color space.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn xyz2lab(c: vec3f) -> vec3f {
    let n = c / vec3f(95.047, 100.0, 108.883);
    let c0 = pow(n, vec3f(1.0 / 3.0));
    let c1 = (7.787 * n) + (16.0 / 116.0);
    let v = mix(c0, c1, step(n, vec3f(0.008856)));
    return vec3f( (116.0 * v.y) - 16.0,
                   500.0 * (v.x - v.y),
                   200.0 * (v.y - v.z) );
}
