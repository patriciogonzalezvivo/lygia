#include "../color/space/w2rgb.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Wavelength to RGB
use: <float3> wavelength(<float> wavelength)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn wavelength(w: f32) -> vec3f { return w2rgb(w); }
