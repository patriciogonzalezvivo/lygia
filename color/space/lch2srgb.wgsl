#include "lch2lab.wgsl"
#include "lab2srgb.wgsl"
/*
contributors: Patricio Gonzalez Vivo
description: "Converts a Lch to sRGB color space. \nNote: LCh is simply Lab but converted to polar coordinates (in degrees).\n"
use: lch2srgb(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn lch2srgb3(lch: vec3f) -> vec3f { return lab2srgb( lch2lab(lch) ); }
fn lch2srgb4(lch: vec4f) -> vec4f { return vec4f(lch2srgb(lch.xyz),lch.a);}
