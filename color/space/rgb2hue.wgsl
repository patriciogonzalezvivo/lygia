/*
contributors:
    - Sam Hocevar
    - Patricio Gonzalez Vivo
description: Convert a color from RGB to HSL color space.
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rgb2hue(rgb: vec3f) -> f32 {
    let K = vec4ff(0.0, -0.33333333333333333333, 0.6666666666666666666, -1.0);
    let p = rgb.g < rgb.b ? vec4f(rgb.bg, K.wz) : vec4f(rgb.gb, K.xy);
    let q = rgb.r < p.x ? vec4f(p.xyw, rgb.r) : vec4f(rgb.r, p.yzx);
    let d = q.x - min(q.w, q.y);
    return abs(q.z + (q.w - q.y) / (6. * d + 1e-10));
}