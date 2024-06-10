#include "rgb2xyz.wgsl"
#include "xyz2lab.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB color to Lab color space.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn rgb2lab(rgb: vec3f ) -> vec3f { return xyz2lab(rgb2xyz(rgb)); }
