/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a linear RGB color to XYZ color space. Based on http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #ifdef CIE_D50
// const mat3 RGB2XYZ = mat3(
//     0.4360747, 0.2225045, 0.0139322,
//     0.3850649, 0.7168786, 0.0971045,
//     0.1430804, 0.0606169, 0.7141733);
// #else
// #endif
// #endif

fn rgb2xyz(rgb: vec3f) -> vec3f { return RGB2XYZ * rgb; }