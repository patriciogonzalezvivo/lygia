#include "triangleNoise.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: Gjøl 2016, "Banding in Games: A Noisy Rant"
use: 
 - <vec4|vec3|float> ditherTriangleNoiseRGB(<vec4|vec3|float> value, <vec2> st, <float> time)
 - <vec4|vec3|float> ditherTriangleNoiseRGB(<vec4|vec3|float> value, <float> time)
*/

#ifndef HIGHP
#if defined(TARGET_MOBILE) && defined(GL_ES)
#define HIGHP highp
#else
#define HIGHP
#endif
#endif

#ifndef DITHER_TRIANGLENOISERGB
#define DITHER_TRIANGLENOISERGB

float ditherTriangleNoiseRGB(const in float b, const HIGHP in vec2 st, const HIGHP in float time) {
    return b + triangleNoise(st, time) / 255.0;
}

vec3 ditherTriangleNoiseRGB(const in vec3 rgb, const HIGHP in vec2 st, const HIGHP in float time) {
    vec3 dither = vec3(
            triangleNoise(st, time),
            triangleNoise(st + 0.1337, time),
            triangleNoise(st + 0.3141, time)) / 255.0;
    return rgb + dither;
}

vec4 ditherTriangleNoiseRGB(const in vec4 rgba, const HIGHP in vec2 st, const HIGHP in float time) {
    vec3 dither = vec3(
            triangleNoise(st, time),
            triangleNoise(st + 0.1337, time),
            triangleNoise(st + 0.3141, time)) / 255.0;
    return vec4(rgba.rgb + dither, rgba.a + dither.x);
}

#if defined(RESOLUTION)
float ditherTriangleNoiseRGB(const in float b, const HIGHP in float time) {
    return ditherTriangleNoiseRGB(b, gl_FragCoord.xy / RESOLUTION, time);
}

vec3 ditherTriangleNoiseRGB(const in vec3 rgb, const HIGHP in float time) {
    return ditherTriangleNoiseRGB(rgb, gl_FragCoord.xy / RESOLUTION, time);
}

vec4 ditherTriangleNoiseRGB(const in vec4 b, const HIGHP in float time) {
    return ditherTriangleNoiseRGB(b, gl_FragCoord.xy / RESOLUTION, time);
}
#endif

#endif