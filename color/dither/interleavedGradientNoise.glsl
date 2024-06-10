#include "../../math/decimate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: 'Jimenez 2014, "Next Generation Post-Processing Call of Duty" http://advances.realtimerendering.com/s2014/index.html'
use: <vec4|vec3|float> interleavedGradientNoise(<vec4|vec3|float> value, <float> time)
options:
    - DITHER_INTERLEAVEDGRADIENTNOISE_TIME
    - DITHER_INTERLEAVEDGRADIENTNOISE_COORD
    - DITHER_INTERLEAVEDGRADIENTNOISE_CHROMATIC
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef DITHER_INTERLEAVEDGRADIENTNOISE_COORD
#define DITHER_INTERLEAVEDGRADIENTNOISE_COORD gl_FragCoord.xy
#endif

#ifdef DITHER_TIME
#define DITHER_INTERLEAVEDGRADIENTNOISE_TIME DITHER_TIME
#endif

#ifndef DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION
#ifdef DITHER_PRECISION
#define DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION DITHER_PRECISION
#else
#define DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION 255
#endif
#endif

#ifndef FNC_DITHER_INTERLEAVEDGRADIENTNOISE
#define FNC_DITHER_INTERLEAVEDGRADIENTNOISE

float ditherInterleavedGradientNoise(vec2 st) {
    #ifdef DITHER_INTERLEAVEDGRADIENTNOISE_TIME
    st += 1337.0*fract(DITHER_INTERLEAVEDGRADIENTNOISE_TIME);
    #endif
    st = floor(st);
    return fract(52.982919 * fract(dot(vec2(0.06711, 0.00584), st))) * 2.0 - 1.0;
}

float ditherInterleavedGradientNoise(const float value, const vec2 st, const int pres) {
    float ditherPattern = ditherInterleavedGradientNoise(st);
    return value + ditherPattern / 255.0;
}

vec3 ditherInterleavedGradientNoise(const vec3 color, const vec2 st, const int pres) {
    #ifdef DITHER_INTERLEAVEDGRADIENTNOISE_CHROMATIC 
    vec3 ditherPattern = vec3(
            ditherInterleavedGradientNoise(st),
            ditherInterleavedGradientNoise(st + 0.1337),
            ditherInterleavedGradientNoise(st + 0.3141));
    #else
    vec3 ditherPattern = vec3(ditherInterleavedGradientNoise(st));
    #endif
    
    // return color + ditherPattern / 255.0;

    float d = float(pres);
    float h = 0.5 / d;
    // vec3 decimated = decimate(color, d);
    // vec3 diff = (color - decimated) * d;
    // ditherPattern = step(ditherPattern, diff);
    return decimate(color - h + ditherPattern / d, d);
}

// float ditherInterleavedGradientNoise(const float b, const vec2 st) { return b + triangleNoise(st) / float(DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION); }
vec3 ditherInterleavedGradientNoise(const vec3 color, const vec2 xy) {  return ditherInterleavedGradientNoise(color, xy, DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION); }
vec4 ditherInterleavedGradientNoise(const vec4 color, const vec2 xy) {  return vec4(ditherInterleavedGradientNoise(color.rgb, xy, DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION), color.a); }

float ditherInterleavedGradientNoise(const float val, int pres) { return ditherInterleavedGradientNoise(vec3(val),DITHER_INTERLEAVEDGRADIENTNOISE_COORD, pres).r; }
vec3 ditherInterleavedGradientNoise(const vec3 color, int pres) { return ditherInterleavedGradientNoise(color, DITHER_INTERLEAVEDGRADIENTNOISE_COORD, pres); }
vec4 ditherInterleavedGradientNoise(const vec4 color, int pres) { return vec4(ditherInterleavedGradientNoise(color.rgb, DITHER_INTERLEAVEDGRADIENTNOISE_COORD, pres), color.a); }

float ditherInterleavedGradientNoise(const float val) { return ditherInterleavedGradientNoise(vec3(val), DITHER_INTERLEAVEDGRADIENTNOISE_COORD, DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION).r; }
vec3 ditherInterleavedGradientNoise(const vec3 color) { return ditherInterleavedGradientNoise(color, DITHER_INTERLEAVEDGRADIENTNOISE_COORD, DITHER_INTERLEAVEDGRADIENTNOISE_PRECISION); }
vec4 ditherInterleavedGradientNoise(const vec4 color) { return vec4(ditherInterleavedGradientNoise(color.rgb), color.a); }

#endif