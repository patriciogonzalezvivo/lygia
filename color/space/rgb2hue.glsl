/*
contributors:
    - Sam Hocevar
    - Patricio Gonzalez Vivo
description: 'Convert a color from RGB to HSL color space.'
use: <float> rgb2hue(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef HUE_EPSILON
#define HUE_EPSILON 1e-10
#endif

#ifndef FNC_RGB2HUE
#define FNC_RGB2HUE
float rgb2hue(const in vec3 c) {
    vec4 K = vec4(0.0, -0.33333333333333333333, 0.6666666666666666666, -1.0);
    vec4 p = c.g < c.b ? vec4(c.bg, K.wz) : vec4(c.gb, K.xy);
    vec4 q = c.r < p.x ? vec4(p.xyw, c.r) : vec4(c.r, p.yzx);
    float d = q.x - min(q.w, q.y);
    return abs(q.z + (q.w - q.y) / (6. * d + HUE_EPSILON));
}

float rgb2hue(const in vec4 c) { return rgb2hue(c.rgb); }
#endif