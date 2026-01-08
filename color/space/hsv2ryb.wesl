import lygia::color::space::{hsv2rgb::hsv2rgb, ryb2rgb::ryb2rgb};
import lygia::math::saturate::saturate3;

/*
contributors: Patricio Gonzalez Vivo
description: Convert from HSV to RYB color space
use: <vec3> hsv2ryb(<vec3> hsv)
note: |
    Three implementations are provided via WESL conditionals:
    - @if(HSV2RYB_FAST): CMY bias version (fastest)
    - @elif(RYB_FAST): Fast RYB conversion via HSV->RGB->RYB
    - @else: Default version with saturation adjustment
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/color_ryb.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

@if(HSV2RYB_FAST)
fn hsv2ryb(v: vec3f) -> vec3f {
    let f = fract(v.x) * 6.0;
    var c = smoothstep(vec3f(3.0, 0.0, 3.0), vec3f(2.0, 2.0, 4.0), vec3f(f));
    c += smoothstep(vec3f(4.0, 3.0, 4.0), vec3f(6.0, 4.0, 6.0), vec3f(f)) * vec3f(1.0, -1.0, -1.0);
    return mix(vec3f(1.0), c, v.y) * v.z;
}
@elif(RYB_FAST)
fn hsv2ryb(v: vec3f) -> vec3f {
    let rgb = hsv2rgb(v);
    return ryb2rgb(rgb);
}
@else
fn hsv2ryb(v: vec3f) -> vec3f {
    let rgb = hsv2rgb(v);
    return ryb2rgb(rgb) - saturate3(vec3f(1.0 - v.z));
}