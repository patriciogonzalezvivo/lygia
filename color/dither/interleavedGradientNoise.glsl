/*
original_author: Patricio Gonzalez Vivo
description: Jimenez 2014, "Next Generation Post-Processing in Call of Duty"
use: <vec4|vec3|float> interleavedGradientNoise(<vec4|vec3|float> value, <float> time)
*/


#ifndef HIGHP
#if defined(TARGET_MOBILE) && defined(GL_ES)
#define HIGHP highp
#else
#define HIGHP
#endif
#endif

#ifndef TIME_SECS
#define TIME_SECS u_time
#endif

#ifndef DITHER_INTERLEAVEDGRADIENTNOISE
#define DITHER_INTERLEAVEDGRADIENTNOISE

float interleavedGradientNoise(const HIGHP in vec2 n) {
    return fract(52.982919 * fract(dot(vec2(0.06711, 0.00584), n)));
}

float ditherInterleavedGradientNoise(float b) {
    float noise = interleavedGradientNoise(gl_FragCoord.xy + TIME_SECS);
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return b + noise / 255.0;
}

vec3 ditherInterleavedGradientNoise(vec3 rgb) {
    float noise = interleavedGradientNoise(gl_FragCoord.xy + TIME_SECS);
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return rgb.rgb + noise / 255.0;
}

vec4 ditherInterleavedGradientNoise(vec4 rgba) {
    return vec4(ditherInterleavedGradientNoise(rgba.rgb), rgba.a);
}

#endif