import package::color::space::rgb2xyz::rgb2xyz;
import package::color::space::xyz2xyY::xyz2xyY;

/*
contributors: Patricio Gonzalez Vivo
description: Converts a linear RGB color to xyY color space.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rgb2xyY(rgb: vec3f) -> vec3f { return xyz2xyY(rgb2xyz(rgb)); }
