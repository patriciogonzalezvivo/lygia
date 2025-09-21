import package::color::space::rgb2hcv::rgb2hcv;

/*
contributors: Patricio Gonzalez Vivo
description: 'Convert from linear RGB to HSL. Based on work by Sam Hocevar and Emil Persson'
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rgb2hsl(rgb: vec3f) -> vec3f {
    let HCV = rgb2hcv(rgb);
    let L = HCV.z - HCV.y * 0.5;
    let S = HCV.y / (1.0 - abs(L * 2.0 - 1.0) + 1e-10);
    return vec3f(HCV.x, S, L);
}
