/*
contributors: Patricio Gonzalez Vivo
description: "Convert from linear RGB to YIQ which was the following range. \nUsing conversion matrices from FCC NTSC Standard (SMPTE C) https://en.wikipedia.org/wiki/YIQ\n"
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rgb2yiq(rgb : vec3f) -> vec3f { return RGB2YIQ * rgb; }