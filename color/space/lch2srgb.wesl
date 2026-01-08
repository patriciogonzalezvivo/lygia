import lygia::color::space::lch2lab::lch2lab;
import lygia::color::space::lab2srgb::lab2srgb;

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a Lch to sRGB color space.
    Note: LCh is simply Lab but converted to polar coordinates (in degrees).
use: lch2srgb(<vec3f|vec4f> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn lch2srgb(lch: vec3f) -> vec3f {
    return lab2srgb(lch2lab(lch));
}

fn lch2srgb4(lch: vec4f) -> vec4f {
    return vec4f(lch2srgb(lch.xyz), lch.a);
}
