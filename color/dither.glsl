/*
original_author: Patricio Gonzalez Vivo
description: set of dither methods
use: <vec4|vec3|float> dither(<vec4|vec3|float> value[, <float> time])
Options:
    - DITHER_FNC
    - TIME_SECS: elapsed time in seconds
    - RESOLUTION: view port resolution
    - BLUENOISE_TEXTURE_RESOLUTION
    - BLUENOISE_TEXTURE
    - DITHER_ANIMATED
    - DITHER_CHROMATIC
    - SAMPLER_FNC
examples:
    - /shaders/color_dither.frag
*/

#include "dither/interleavedGradientNoise.glsl"
#include "dither/vlachos.glsl"
#include "dither/triangleNoise.glsl"
#include "dither/blueNoise.glsl"
#include "dither/shift.glsl"

#ifndef DITHER_FNC
#ifdef TARGET_MOBILE
#define DITHER_FNC ditherInterleavedGradientNoise
#else
#define DITHER_FNC ditherVlachos
#endif
#endif

#ifndef FNC_DITHER
#define FNC_DITHER

float dither(float v, const HIGHP in float time) { return DITHER_FNC(v, time); }
vec3 dither(vec3 v, const HIGHP in float time) { return DITHER_FNC(v, time); }
vec4 dither(vec4 v, const HIGHP in float time) { return DITHER_FNC(v, time); }

#ifdef TIME_SECS
float dither(float v) { return dither(v, TIME_SECS); }
vec3 dither(vec3 v) { return dither(v, TIME_SECS); }
vec4 dither(vec4 v) { return dither(v, TIME_SECS); }
#endif

#endif