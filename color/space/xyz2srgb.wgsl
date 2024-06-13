#include "xyz2rgb.wgsl"
#include "rgb2srgb.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a XYZ color to sRGB. From http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn xyz2srgb(xyz: vec3f) -> vec3f { return rgb2srgb(xyz2rgb(xyz)); }