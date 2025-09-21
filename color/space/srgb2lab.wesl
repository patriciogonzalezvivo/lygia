import package::color::space::{rgb2lab::rgb2lab, srgb2rgb::srgb2rgb};

/*
contributors: Patricio Gonzalez Vivo
description: Converts a sRGB color to Lab
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn srgb2lab(srgb: vec3f) -> vec3f { return rgb2lab(srgb2rgb(srgb));}
