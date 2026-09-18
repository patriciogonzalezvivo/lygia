/*
contributors: Patricio Gonzalez Vivo
description: "Convert LST to RGB. LMS (long, medium, short), is a color space which\
    \ represents the response of the three types of cones of the human eye, named for\
    \ their responsivity (sensitivity) peaks at long, medium, and short wavelengths.\
    \ \nRefs https://en.wikipedia.org/wiki/LMS_color_space https://arxiv.org/pdf/1711.10662\n"
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// const LMS2RGB = mat3x3<f32>(
//     vec3f(2.85847e+0, -1.62879e+0, -2.48910e-2),
//     vec3f(-2.10182e-1,  1.15820e+0,  3.24281e-4),
//     vec3f(-4.18120e-2, -1.18169e-1,  1.06867e+0)
// );

);

fn lms2rgb(lms : vec3f) -> vec3f { return LMS2RGB * lms; }
