import lygia::color::space::{lab2xyz::lab2xyz, xyz2rgb::xyz2rgb};

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a Lab color to RGB color space. https://en.wikipedia.org/wiki/CIELAB_color_space
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn lab2rgb(lab: vec3f) -> vec3f { return xyz2rgb( lab2xyz( lab ) ); }
