/*
contributors: Patricio Gonzalez Vivo
description: Gets the luminosity from linear RGB, based on Rec709 luminance (see https://en.wikipedia.org/wiki/Grayscale)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rgb2luma(rgb: vec3f) -> f32 { return dot(rgb, vec3(0.2126, 0.7152, 0.0722)); }