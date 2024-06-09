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

#ifndef FNC_HSV2RYB
#define FNC_HSV2RYB

vec3 hsv2ryb( in vec3 v ) {
    float f = fract(v.x) * 6.0;
    vec3 c = smoothstep(vec3(3.,0.,3.),vec3(2.,2.,4.), vec3(f));
    c += smoothstep(vec3(4.,3.,4.),vec3(6.,4.,6.), vec3(f)) * vec3(1., -1., -1.);
    return mix(vec3(1.), c, v.y) * v.z;
}

#endif