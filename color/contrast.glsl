/*
contributors: Patricio Gonzalez Vivo
description: Bias high pass
use: <vec4|vec3|float> contrast(<vec4|vec3|float> value, <float> amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_CONTRAST
#define FNC_CONTRAST
float contrast(in float v, in float a) { return (v - 0.5 ) * a + 0.5; }
vec3 contrast(in vec3 v, in float a) { return (v - 0.5 ) * a + 0.5; }
vec4 contrast(in vec4 v, in float a) { return vec4((v.rgb - 0.5 ) * a + 0.5, v.a); }
#endif
