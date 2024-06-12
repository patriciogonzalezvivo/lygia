/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a Lab color to XYZ color space. https://en.wikipedia.org/wiki/CIELAB_color_space'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn lab2xyz(c : vec3f) -> vec3f {
    var f = vec3f(0.0);
    f.y = (c.x + 16.0) / 116.0;
    f.x = c.y / 500.0 + f.y;
    f.z = f.y - c.z / 200.0;
    let c0 = f * f * f;
    let c1 = (f - 16.0 / 116.0) / 7.787;
    return vec3f(95.047, 100.000, 108.883) * mix(c0, c1, step(f, vec3f(0.206897)));
}
