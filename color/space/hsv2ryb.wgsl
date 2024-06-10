/*
contributors: Patricio Gonzalez Vivo
description: Convert from HSV to RYB color space
use: <vec3> hsv2ryb(<vec3> hsv)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn hsv2ryb(v: vec3f) -> vec3f {
    let f = fract(v.x) * 6.0;
    var c = smoothstep(vec3f(3.,0.,3.),vec3f(2.,2.,4.), vec3f(f));
    c += smoothstep(vec3f(4.,3.,4.),vec3f(6.,4.,6.), vec3f(f)) * vec3f(1., -1., -1.);
    return mix(vec3f(1., 1., 1.), c, v.y) * v.z;
}