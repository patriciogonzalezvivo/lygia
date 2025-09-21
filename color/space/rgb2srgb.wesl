/*
contributors: Patricio Gonzalez Vivo
description: Converts a linear RGB color to sRGB.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rgb2srgb_mono(channel: f32) -> f32 {
    if (channel < 0.0031308) {
        return 12.92 * channel;
    }
    else {
        return (1.055) * pow(channel, 0.4166666666666667) - 0.055;
    }
}

fn rgb2srgb(rgb: vec3f) -> vec3f {
    return saturate(vec3(rgb2srgb_mono(rgb.r), rgb2srgb_mono(rgb.g), rgb2srgb_mono(rgb.b)));
}
