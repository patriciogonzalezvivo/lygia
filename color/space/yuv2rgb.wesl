/*
contributors: Patricio Gonzalez Vivo
description: Pass a color in YUB and get RGB color
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


// #ifdef YUV_SDTV
// const YUV2RGB = mat3x3<f32>(
//     vec3f(1.0,       1.0,      1.0),
//     vec3f(0.0,      -0.39465,  2.03211),
//     vec3f(1.13983,  -0.58060,  0.0)
// );
// #else
const YUV2RGB = mat3x3<f32>(
    vec3f(1.0,       1.0,      1.0),
    vec3f(0.0,      -0.21482,  2.12798),
    vec3f(1.28033,  -0.38059,  0.0)
);
// #endif
fn yuv2rgb(yuv: vec3f) -> vec3f { return YUV2RGB * yuv; }

