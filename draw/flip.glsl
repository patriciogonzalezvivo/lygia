/*
contributors: Patricio Gonzalez Vivo
description: Flips the float passed in, 0 becomes 1 and 1 becomes 0
use: flip(<float> v, <float> pct)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_FLIP
#define FNC_FLIP
float flip(in float v, in float pct) {
    return mix(v, 1.0 - v, pct);
}

vec3 flip(in vec3 v, in float pct) {
    return mix(v, 1.0 - v, pct);
}

vec4 flip(in vec4 v, in float pct) {
    return mix(v, 1.0 - v, pct);
}
#endif
