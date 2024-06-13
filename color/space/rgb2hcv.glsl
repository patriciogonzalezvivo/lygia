/*
contributors: Patricio Gonzalez Vivo
description: 'Convert from RGB to HCV (Hue, Chroma, Value). Based on work by Sam Hocevar and Emil Persson'
use: rgb2xyz(<vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef HCV_EPSILON
#define HCV_EPSILON 1e-10
#endif

#ifndef FNC_RGB2HCV
#define FNC_RGB2HCV
vec3 rgb2hcv(const in vec3 rgb) {
    vec4 P = (rgb.g < rgb.b) ? vec4(rgb.bg, -1.0, 2.0/3.0) : vec4(rgb.gb, 0.0, -1.0/3.0);
    vec4 Q = (rgb.r < P.x) ? vec4(P.xyw, rgb.r) : vec4(rgb.r, P.yzx);
    float C = Q.x - min(Q.w, Q.y);
    float H = abs((Q.w - Q.y) / (6.0 * C + HCV_EPSILON) + Q.z);
    return vec3(H, C, Q.x);
}
vec4 rgb2hcv(vec4 rgb) {return vec4(rgb2hcv(rgb.rgb), rgb.a);}
#endif