#include "stroke.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Create a bridge on a given in_value and draw a stroke inside that gap
use: bridge(<float|vec2|vec3|vec4> in_value, <float> sdf, <float> size, <float> width)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn bridge(c: f32, d: f32, s: f32, w: f32) -> f32 {
    c *= 1.0 - stroke(d, s , w * 2.0);
    return c + stroke(d, s, w);
}

fn bridge2(c: vec2f, d: f32, s: f32, w: f32) -> vec2f {
    c *= 1.0 - stroke(d, s , w * 2.0);
    return c + stroke(d, s, w);
}

fn bridge3(c: vec3f, d: f32, s: f32, w: f32) -> vec3f {
    c *= 1.0 - stroke(d, s , w * 2.0);
    return c + stroke(d, s, w);
}

fn bridge4(c: vec4f, d: f32, s: f32, w: f32) -> vec4f {
    c *= 1.0 - stroke(d, s , w * 2.0);
    return c + stroke(d, s, w);
}
