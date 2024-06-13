/*
contributors: Patricio Gonzalez Vivo
description: Get's the luminosity from sRGB. Based on from Rec601 luma (see https://en.wikipedia.org/wiki/Grayscale)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn srgb2luma(srgb : vec3f) -> f32 { return dot(srgb, vec3f(0.299, 0.587, 0.114)); }