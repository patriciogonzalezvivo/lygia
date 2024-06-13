/*
contributors: Patricio Gonzalez Vivo
description: "Converts a color in YIQ to linear RGB color. \nFrom https://en.wikipedia.org/wiki/YIQ\n"
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const YIQ2RGB : mat3x3<f32>  = mat3x3<f32>( 
    vec3f(1.0,  0.9469,  0.6235), 
    vec3f(1.0, -0.2747, -0.6357), 
    vec3f(1.0, -1.1085,  1.7020) );
fn yiq2rgb(yiq : vec3f) -> vec3f { return YIQ2RGB * yiq; }