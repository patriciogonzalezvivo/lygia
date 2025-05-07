#include "xyz2srgb.wgsl"
#include "xyY2xyz.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts from xyY to sRGB
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn xyY2srgb(xyY: vec3f) -> vec3f { return xyz2srgb(xyY2xyz(xyY)); }