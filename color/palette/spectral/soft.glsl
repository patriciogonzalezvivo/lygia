#include "../../../math/const.glsl"
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

#ifndef FNC_CHROMA
#define FNC_CHROMA

vec3 spectral_soft(float x) {
    float delta = 0.5;
    vec3 color = vec3(1.0);
    float freq = x * PI;
    color.r = sin(freq - delta);
    color.g = sin(freq);
    color.b = sin(freq + delta);
    return pow(color, vec3(4.0));
}

#endif