/*
original_author: Patricio Gonzalez Vivo
description: set of dither methods
use: <vec4|vec3|float> dither(<vec4|vec3|float> value[, <float> time])
Options:
    - TIME_SECS: elapsed time in seconds
    - RESOLUTION: view port resolution
*/

#ifndef TIME_SECS
#ifdef GLSLVIEWER
#define TIME_SECS u_time
#endif 
#endif

#ifndef RESOLUTION
#ifdef GLSLVIEWER
#define RESOLUTION u_resolution
#endif
#endif

#include "dither/interleavedGradientNoise.glsl"
#include "dither/vlachos.glsl"
#include "dither/triangleNoise.glsl"
#include "dither/triangleNoiseRGB.glsl"

#ifndef DITHER_FNC
#ifdef TARGET_MOBILE
#define DITHER_FNC ditherInterleavedGradientNoise
#else
#define DITHER_FNC ditherVlachos
#endif
#endif

#ifndef FNC_DITHER
#define FNC_DITHER

float dither(float b, const HIGHP in float time) {
    return DITHER_FNC(b, time);
}

vec3 dither(vec3 rgb, const HIGHP in float time) {
    return DITHER_FNC(rgb, time);
}

vec4 dither(vec4 rgba, const HIGHP in float time) {
    return DITHER_FNC(rgba, time);
}

#ifdef TIME_SECS
float dither(float b) {
    return DITHER_FNC(b, TIME_SECS);
}

vec3 dither(vec3 rgb) {
    return DITHER_FNC(rgb, TIME_SECS);
}

vec4 dither(vec4 rgba) {
    return DITHER_FNC(rgba, TIME_SECS);
}
#endif

#endif