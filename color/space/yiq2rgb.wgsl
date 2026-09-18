/*
contributors: Patricio Gonzalez Vivo
description: "Converts a color in YIQ to linear RGB color. \nFrom https://en.wikipedia.org/wiki/YIQ\n"
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn yiq2rgb(yiq : vec3f) -> vec3f { return YIQ2RGB * yiq; }