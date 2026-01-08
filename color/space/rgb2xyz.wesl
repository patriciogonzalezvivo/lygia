/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a linear RGB color to XYZ color space. Based on http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(CIE_D50)
const RGB2XYZ = mat3x3f(
     0.4360747, 0.2225045, 0.0139322,
     0.3850649, 0.7168786, 0.0971045,
     0.1430804, 0.0606169, 0.7141733);

@if(!CIE_D50)
const RGB2XYZ = mat3x3f(
    0.4124564, 0.2126729, 0.0193339,
    0.3575761, 0.7151522, 0.1191920,
    0.1804375, 0.0721750, 0.9503041 );

fn rgb2xyz(rgb: vec3f) -> vec3f {
    // Scale to 0-100 range (colorimetry standard, different from GLSL, see #271)
    return RGB2XYZ * rgb * 100.0;
}

fn rgb2xyz4(color: vec4f) -> vec4f {
    return vec4f(rgb2xyz(color.rgb), color.a);
}
