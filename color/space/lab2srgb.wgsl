#include "lab2xyz.wgsl"
#include "xyz2srgb.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a Lab color to RGB color space. https://en.wikipedia.org/wiki/CIELAB_color_space'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn lab2srgb(lab: vec3f) -> vec3f { return xyz2srgb( lab2xyz( lab ) ); }
