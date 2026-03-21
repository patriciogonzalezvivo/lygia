#include "../color/space/k2rgb.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Blackbody in kelvin to RGB. Range between 0.0 and 40000.0 Kelvin
use: <vec3> blackbody(<float> wavelength)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn blackbody(k: f32) -> vec3f { return k2rgb(k); }
