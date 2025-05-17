import lygia::color::space::{lch2lab::lch2lab, lab2rgb::lab2rgb};

/*
contributors: Patricio Gonzalez Vivo
description: "Converts a Lch to linear RGB color space. \nNote: LCh is simply Lab but converted to polar coordinates (in degrees).\n"
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn lch2rgb(lch: vec3f) -> vec3f { return lab2rgb( lch2lab(lch) ); }
