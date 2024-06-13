/*
contributors: Patricio Gonzalez Vivo
description: Converts YCbCr to RGB according to https://en.wikipedia.org/wiki/YCbCr
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn YCbCr2rgb(ycbcr: vec3f) -> vec3f {
    let cb = ycbcr.y - 0.5;
    let cr = ycbcr.z - 0.5;
    let y = ycbcr.x;
    let r = 1.402 * cr;
    let g = -0.344 * cb - 0.714 * cr;
    let b = 1.772 * cb;
    return vec3f(r, g, b) + y;
}
