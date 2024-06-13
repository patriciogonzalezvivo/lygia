#include "rgb2hcv.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Convert from linear RGB to HSL. Based on work by Sam Hocevar and Emil Persson'
use: <vec3|vec4> rgb2hsl(<vec3|vec4> rgb)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef HSL_EPSILON
#define HSL_EPSILON 1e-10
#endif

#ifndef FNC_RGB2HSL
#define FNC_RGB2HSL
vec3 rgb2hsl(const in vec3 rgb) {
    vec3 HCV = rgb2hcv(rgb);
    float L = HCV.z - HCV.y * 0.5;
    float S = HCV.y / (1.0 - abs(L * 2.0 - 1.0) + HSL_EPSILON);
    return vec3(HCV.x, S, L);
}
vec4 rgb2hsl(const in vec4 rgb) { return vec4(rgb2hsl(rgb.xyz),rgb.a);}
#endif