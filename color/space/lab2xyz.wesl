/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a Lab color to XYZ color space. https://en.wikipedia.org/wiki/CIELAB_color_space'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// LATER make this a @param const when WESL supports that
@if(CIE_D50)
const CIE_WEIGHT: vec3f = vec3f(0.96429567643, 1.0, 0.82510460251);
@else
// D65
const CIE_WEIGHT: vec3f = vec3f(0.95045592705, 1.0, 1.08905775076);

fn lab2xyz(c : vec3f) -> vec3f {
    var f = vec3f(0.0);
    f.y = (c.x + 16.0) / 116.0;
    f.x = c.y / 500.0 + f.y;
    f.z = f.y - c.z / 200.0;
    let c0 = f * f * f;
    let c1 = (f - 16.0 / 116.0) / 7.787;
    // Scale to 0-100 range (colorimetry standard)
    return CIE_WEIGHT * 100.0 * mix(c0, c1, step(f, vec3f(0.206897)));
}

fn lab2xyz4(c: vec4f) -> vec4f {
    return vec4f(lab2xyz(c.xyz), c.a);
}
