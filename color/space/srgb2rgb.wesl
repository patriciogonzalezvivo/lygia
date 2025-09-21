/*
contributors: Patricio Gonzalez Vivo
description: sRGB to linear RGB conversion.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn srgb2rgb_mono(channel: f32) -> f32 {
    if (channel < 0.04045) {
        return channel * 0.0773993808;
    }
    else {
        return pow((channel + 0.055) * 0.947867298578199, 2.4);
    }
}

fn srgb2rgb(srgb:vec3f) -> vec3f {
    return vec3f(
            srgb2rgb_mono(srgb.r + 0.00000001),
            srgb2rgb_mono(srgb.g + 0.00000001),
            srgb2rgb_mono(srgb.b + 0.00000001)
        );
}
