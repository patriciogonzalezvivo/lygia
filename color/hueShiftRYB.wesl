import lygia::color::hueShift::hueShift;
import lygia::color::space::{rgb2ryb::rgb2ryb, ryb2rgb::ryb2rgb};

/*
contributors:
    - Johan Ismael
    - Patricio Gonzalez Vivo
description: Shifts color hue in the RYB color space
use: hueShiftRYB(<vec3f|vec4f> color, <f32> angle)
note: |
    This function internally uses hueShift, which supports both angle and normalized modes.
    See hueShift.wesl for the HUESHIFT_AMOUNT conditional implementation.
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn hueShiftRYB(color: vec3f, a: f32) -> vec3f {
    var rgb = rgb2ryb(color);
    rgb = hueShift(rgb, a);
    return ryb2rgb(rgb);
}

fn hueShiftRYB4(color: vec4f, a: f32) -> vec4f {
    return vec4f(hueShiftRYB(color.rgb, a), color.a);
}
