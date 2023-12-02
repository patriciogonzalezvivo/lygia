/*
contributors: Patricio Gonzalez Vivo
description: |
    Jimenez 2014, "Next Generation Post-Processing in Call of Duty" http://advances.realtimerendering.com/s2014/index.html
use: <vec4|vec3|float> interleavedGradientNoise(<vec4|vec3|float> value, <float> time)
options:
    - DITHER_INTERLEAVEDGRADIENTNOISE_TIME
examples:
    - /shaders/color_dither.frag
*/

#ifndef DITHER_INTERLEAVEDGRADIENTNOISE_COORD
#define DITHER_INTERLEAVEDGRADIENTNOISE_COORD gl_FragCoord.xy
#endif

#ifdef DITHER_TIME
#define DITHER_INTERLEAVEDGRADIENTNOISE_TIME DITHER_TIME
#endif

#ifndef FNC_DITHER_INTERLEAVEDGRADIENTNOISE
#define FNC_DITHER_INTERLEAVEDGRADIENTNOISE

float ditherInterleavedGradientNoise(vec2 n) {
    return fract(52.982919 * fract(dot(vec2(0.06711, 0.00584), n)));
}

float ditherInterleavedGradientNoise(float b) {
    vec2 st = DITHER_INTERLEAVEDGRADIENTNOISE_COORD;
    #ifdef DITHER_INTERLEAVEDGRADIENTNOISE_TIME
    st += 1337.0*fract(DITHER_INTERLEAVEDGRADIENTNOISE_TIME);
    #endif
    float noise = ditherInterleavedGradientNoise(st);
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return b + noise / 255.0;
}

vec3 ditherInterleavedGradientNoise(vec3 rgb) {
    vec2 st = DITHER_INTERLEAVEDGRADIENTNOISE_COORD;
    #ifdef DITHER_INTERLEAVEDGRADIENTNOISE_TIME
    st += 1337.0*fract(DITHER_INTERLEAVEDGRADIENTNOISE_TIME);
    #endif
    float noise = ditherInterleavedGradientNoise(st);
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return rgb.rgb + noise / 255.0;
}

vec4 ditherInterleavedGradientNoise(vec4 rgba) {
    return vec4(ditherInterleavedGradientNoise(rgba.rgb), rgba.a);
}

#endif