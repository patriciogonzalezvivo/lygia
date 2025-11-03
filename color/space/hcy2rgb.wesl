import lygia::color::space::hue2rgb::hue2rgb;

/*
contributors: Patricio Gonzalez Vivo
description: Converts a HCY color to linear RGB
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn hcy2rgb(hcy_in: vec3f) -> vec3f {
    var hcy = hcy_in;
    let HCYwts = vec3f(0.299, 0.587, 0.114);
    let RGB = hue2rgb(hcy.x);
    let Z = dot(RGB, HCYwts);
    if (hcy.z < Z) {
        hcy.y *= hcy.z / Z;
    } else if (Z < 1.0) {
        hcy.y *= (1.0 - hcy.z) / (1.0 - Z);
    }
    return (RGB - Z) * hcy.y + hcy.z;
}

fn hcy2rgb4(hcy: vec4f) -> vec4f {
    return vec4f(hcy2rgb(hcy.rgb), hcy.a);
}
