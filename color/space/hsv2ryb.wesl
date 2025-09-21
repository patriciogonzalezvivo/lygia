import package::color::space::{hsv2rgb::hsv2rgb, ryb2rgb::ryb2rgb};

/*
contributors: Patricio Gonzalez Vivo
description: Convert from HSV to RYB color space
use: <vec3> hsv2ryb(<vec3> hsv)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn hsv2ryb(v: vec3f) -> vec3f {
    let rgb = hsv2rgb(v);
    return ryb2rgb(rgb) - saturate(1.-v.z);
}