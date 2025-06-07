import lygia::color::space::hue2rgb;

/*
contributors: Patricio Gonzalez Vivo
description: Converts a HSL color to linear RGB
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/


fn hsl2rgb(hsl: vec3f) -> vec3f {
    let rgb = hue2rgb(hsl.x);
    let C = (1.0 - abs(2.0 * hsl.z - 1.0)) * hsl.y;
    return (rgb - 0.5) * C + hsl.z;
}