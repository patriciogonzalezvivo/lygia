/*
contributors: Patricio Gonzalez Vivo
description: sRGB to linear RGB conversion.
use: srgb2rgb(<vec3f|vec4f> srgb)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// LATER make this a @param const when WESL supports that
const SRGB_EPSILON: f32 = 1e-10;

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
            srgb2rgb_mono(srgb.r + SRGB_EPSILON),
            srgb2rgb_mono(srgb.g + SRGB_EPSILON),
            srgb2rgb_mono(srgb.b + SRGB_EPSILON)
        );
}

fn srgb2rgb4(srgb: vec4f) -> vec4f {
    return vec4f(srgb2rgb(srgb.rgb), srgb.a);
}
