/*
contributors: Patricio Gonzalez Vivo
description: Convert from gamma to linear color space.
use: gamma2linear(<float|vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#if !defined(GAMMA) && !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
#define GAMMA 2.2
#endif

#ifndef FNC_GAMMA2LINEAR
#define FNC_GAMMA2LINEAR
float gamma2linear(const in float v) {
#ifdef GAMMA
    return pow(v, GAMMA);
#else
    // assume gamma 2.0
    return v * v;
#endif
}

vec3 gamma2linear(const in vec3 v) {
#ifdef GAMMA
    return pow(v, vec3(GAMMA));
#else
    // assume gamma 2.0
    return v * v;
#endif
}

vec4 gamma2linear(const in vec4 v) {
    return vec4(gamma2linear(v.rgb), v.a);
}
#endif
