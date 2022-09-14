/*
original_author: Patricio Gonzalez Vivo
description: convert from gamma to linear color space.
use: gamma2linear(<float|vec3|vec4> color)
*/

#if !defined(GAMMA) && !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
#define GAMMA 2.2
#endif

#ifndef FNC_GAMMA2LINEAR
#define FNC_GAMMA2LINEAR
float gamma2linear(in float v) {
#ifdef GAMMA
    return pow(v, GAMMA);
#else
    // assume gamma 2.0
    return v * v;
#endif
}

vec3 gamma2linear(in vec3 v) {
#ifdef GAMMA
    return pow(v, vec3(GAMMA));
#else
    // assume gamma 2.0
    return v * v;
#endif
}

vec4 gamma2linear(in vec4 v) {
    return vec4(gamma2linear(v.rgb), v.a);
}
#endif
