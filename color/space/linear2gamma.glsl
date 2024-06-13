/*
contributors: Patricio Gonzalez Vivo
description: Convert from linear to gamma color space.
use: linear2gamma(<float|vec3|vec4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#if !defined(GAMMA) && !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
#define GAMMA 2.2
#endif

#ifndef FNC_LINEAR2GAMMA
#define FNC_LINEAR2GAMMA
vec3 linear2gamma(const in vec3 v) {
#ifdef GAMMA
    return pow(v, vec3(1.0/GAMMA));
#else
    // assume gamma 2.0
    return sqrt(v);
#endif
}

vec4 linear2gamma(const in vec4 v) {
    return vec4(linear2gamma(v.rgb), v.a);
}

float linear2gamma(const in float v) {
#ifdef GAMMA
    return pow(v, 1.0/GAMMA);
#else
    // assume gamma 2.0
    return sqrt(v);
#endif
}
#endif
