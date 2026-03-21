#include "../../../math/const.wgsl"
/*
contributors: Patricio Gonzalez Vivo
description: Soft chroma spectrum
use: <vec3> spectral_soft(<float> value)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_wavelength.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn spectral_soft(x: f32) -> vec3f {
    let delta = 0.5;
    let color = vec3f(1.0);
    let freq = x * PI;
    color.r = sin(freq - delta);
    color.g = sin(freq);
    color.b = sin(freq + delta);
    return pow(color, vec3f(4.0));
}
