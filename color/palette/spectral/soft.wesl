/*
contributors: Patricio Gonzalez Vivo
description: Attempt chroma spectrum
use: <vec3f> spectral_soft(<f32> value)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wavelength.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

import lygia::math::consts::PI;

fn spectral_soft(x: f32) -> vec3f {
    let delta = 0.5;
    let freq = x * PI;
    var color = vec3f(1.0);
    color.r = sin(freq - delta);
    color.g = sin(freq);
    color.b = sin(freq + delta);
    return pow(color, vec3f(4.0));
}
