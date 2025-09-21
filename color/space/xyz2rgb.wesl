/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a XYZ color to linear RGB. From http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const XYZ2RGB = mat3x3<f32>( vec3f( 3.2404542, -0.9692660,  0.0556434),
                             vec3f(-1.5371385,  1.8760108, -0.2040259),
                             vec3f(-0.4985314,  0.0415560,  1.0572252) );

fn xyz2rgb(xyz: vec3f) -> vec3f { return XYZ2RGB * (xyz * 0.01); }
