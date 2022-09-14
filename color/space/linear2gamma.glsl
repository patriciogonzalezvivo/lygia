/*
original_author: Patricio Gonzalez Vivo
description: convert from linear to gamma color space.
use: linear2gamma(<float|vec3|vec4> color)
*/

#if !defined(GAMMA) && !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
#define GAMMA 2.2
#endif

#ifndef FNC_LINEAR2GAMMA
#define FNC_LINEAR2GAMMA
vec3 linear2gamma(in vec3 v) {
#ifdef GAMMA
    return pow(v, vec3(1. / GAMMA));
#else
    // assume gamma 2.0
    return sqrt(v);
#endif
}

vec4 linear2gamma(in vec4 v) {
    return vec4(linear2gamma(v.rgb), v.a);
}

float linear2gamma(in float v) {
#ifdef GAMMA
    return pow(v, 1. / GAMMA);
#else
    // assume gamma 2.0
    return sqrt(v);
#endif
}
#endif
