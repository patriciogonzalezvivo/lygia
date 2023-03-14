/*
original_author: Patricio Gonzalez Vivo
description: |
    Jimenez 2014, "Next Generation Post-Processing in Call of Duty" http://advances.realtimerendering.com/s2014/index.html
use: <vec4|vec3|float> interleavedGradientNoise(<vec4|vec3|float> value, <float> time)
options:
    - DITHER_INTERLEAVEDGRADIENTNOISE_ANIMATED
examples:
    - /shaders/color_dither.frag
*/


#ifndef HIGHP
#if defined(TARGET_MOBILE) && defined(GL_ES)
#define HIGHP highp
#else
#define HIGHP
#endif
#endif

#ifdef DITHER_ANIMATED
#define DITHER_INTERLEAVEDGRADIENTNOISE_ANIMATED
#endif

#ifndef DITHER_INTERLEAVEDGRADIENTNOISE
#define DITHER_INTERLEAVEDGRADIENTNOISE

float interleavedGradientNoise(const HIGHP in vec2 n) {
    return fract(52.982919 * fract(dot(vec2(0.06711, 0.00584), n)));
}

float ditherInterleavedGradientNoise(float b, const HIGHP in float time) {
    vec2 st = gl_FragCoord.xy;
    #ifdef DITHER_INTERLEAVEDGRADIENTNOISE_ANIMATED
    st += 1337.0*fract(time);
    #endif
    float noise = interleavedGradientNoise(st);
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return b + noise / 255.0;
}

vec3 ditherInterleavedGradientNoise(vec3 rgb, const HIGHP in float time) {
    vec2 st = gl_FragCoord.xy;
    #ifdef DITHER_INTERLEAVEDGRADIENTNOISE_ANIMATED
    st += 1337.0*fract(time);
    #endif
    float noise = interleavedGradientNoise(st);
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return rgb.rgb + noise / 255.0;
}

vec4 ditherInterleavedGradientNoise(vec4 rgba, const HIGHP in float time) {
    return vec4(ditherInterleavedGradientNoise(rgba.rgb, time), rgba.a);
}

#endif