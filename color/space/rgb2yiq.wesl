/*
contributors: Patricio Gonzalez Vivo
description: "Convert from linear RGB to YIQ which was the following range. \nUsing conversion matrices from FCC NTSC Standard (SMPTE C) https://en.wikipedia.org/wiki/YIQ\n"
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const RGB2YIQ : mat3x3<f32>  = mat3x3<f32>(
    vec3f(0.300,  0.5900,  0.1100),
    vec3f(0.599, -0.2773, -0.3217),
    vec3f(0.213, -0.5251,  0.3121) );

fn rgb2yiq(rgb : vec3f) -> vec3f { return RGB2YIQ * rgb; }
