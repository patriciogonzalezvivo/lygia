/*
contributors: Patricio Gonzalez Vivo
description: Pass a color in RGB and get it in YPbPr from http://www.equasys.de/colorconversion.html
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// YPBPR_SDTV
// const YPBPR2RGB = mat3x3<f32>( 
//     vec3f(1.0,     1.0,      1.0),
//     vec3f(0.0,    -0.344,    1.772),
//     vec3f(1.402,  -0.714,    0.0)
// );

const YPBPR2RGB = mat3x3<f32>( 
    vec3f(1.0,     1.0,      1.0),
    vec3f(0.0,    -0.187,    1.856),
    vec3f(1.575,  -0.468,    0.0)
);
fn YPbPr2rgb(rgb: vec3f) -> vec3f { return YPBPR2RGB * rgb; }
