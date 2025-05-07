import lygia::color::hueShift;
import lygia::color::space::{rgb2ryb::rgb2ryb, ryb2rgb::ryb2rgb};

/*
contributors:
    - Johan Ismael
    - Patricio Gonzalez Vivo
description: Shifts color hue in the RYB color space
use: hueShift(<vec3|vec4> color, <float> angle)
optionas:
    - HUESHIFT_AMOUNT: if defined, it uses a normalized value instead of an angle
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn hueShiftRYB(color: vec3f, a: f32) -> vec3f {
    var rgb = rgb2ryb(color);
    rgb = hueShift(rgb, PI);
    return ryb2rgb(rgb);
}
