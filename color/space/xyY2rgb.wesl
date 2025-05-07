import lygia::color::space::{xyz2rgb::xyz2rgb, xyY2xyz::xyY2xyz};

/*
contributors: Patricio Gonzalez Vivo
description: Converts from xyY to linear RGB
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn xyY2rgb(xyY: vec3f) -> vec3f { return xyz2rgb(xyY2xyz(xyY)); }