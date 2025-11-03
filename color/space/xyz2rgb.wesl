/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a XYZ color to linear RGB. From http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(CIE_D50)
const XYZ2RGB = mat3x3<f32>( vec3f( 3.1338561, -0.9787684,  0.0719453),
                             vec3f(-1.6168667,  1.9161415, -0.2289914),
                             vec3f(-0.4906146,  0.0334540,  1.4052427) );

@if(!CIE_D50)
// CIE D65
const XYZ2RGB = mat3x3<f32>( vec3f( 3.2404542, -0.9692660,  0.0556434),
                             vec3f(-1.5371385,  1.8760108, -0.2040259),
                             vec3f(-0.4985314,  0.0415560,  1.0572252) );

fn xyz2rgb(xyz: vec3f) -> vec3f {
    // XYZ is 0-100 scale, divide by 100 to get RGB in 0-1 scale
    return XYZ2RGB * (xyz * 0.01);
}

fn xyz2rgb4(color: vec4f) -> vec4f {
    return vec4f(xyz2rgb(color.rgb), color.a);
}
