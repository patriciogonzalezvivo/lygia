import package::color::space::rgb2lab::rgb2lab;
import package::color::space::lab2lch::lab2lch;

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB color to LCh color space.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rgb2lch(rgb: vec3f) -> vec3f { return lab2lch(rgb2lab(rgb)); }