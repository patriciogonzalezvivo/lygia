/*
original_author: Patricio Gonzalez Vivo
description: |
    Gjøl 2016, "Banding in Games: A Noisy Rant"
use: 
 - <vec4|vec3|float> ditherTriangleNoise(<vec4|vec3|float> value, <vec2> st, <float> time)
 - <vec4|vec3|float> ditherTriangleNoise(<vec4|vec3|float> value, <float> time)
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
#define DITHER_TRIANGLENOISE_ANIMATED
#endif

#ifdef DITHER_CHROMATIC
#define DITHER_TRIANGLENOISE_CHROMATIC
#endif


#ifndef DITHER_TRIANGLENOISE
#define DITHER_TRIANGLENOISE

float triangleNoise(HIGHP in vec2 n, const HIGHP in float time) {
    // triangle noise, in [-1.0..1.0[ range
    #ifdef DITHER_TRIANGLENOISE_ANIMATED
    n += vec2(0.07 * fract(time));
    #endif
    n  = fract(n * vec2(5.3987, 5.4421));
    n += dot(n.yx, n.xy + vec2(21.5351, 14.3137));

    HIGHP float xy = n.x * n.y;
    // compute in [0..2[ and remap to [-1.0..1.0[
    return fract(xy * 95.4307) + fract(xy * 75.04961) - 1.0;
}

float ditherTriangleNoise(const in float b, const HIGHP in vec2 st, const HIGHP in float time) {
    return b + triangleNoise(st, time) / 255.0;
}

vec3 ditherTriangleNoise(const in vec3 rgb, const HIGHP in vec2 st, const HIGHP in float time) {
    
    #ifdef DITHER_TRIANGLENOISE_CHROMATIC 
    vec3 dither = vec3(
            triangleNoise(st, time),
            triangleNoise(st + 0.1337, time),
            triangleNoise(st + 0.3141, time));
    #else
    vec3 dither = vec3(triangleNoise(st, time));
    #endif
            
    return rgb + dither / 255.0;
}

vec4 ditherTriangleNoise(const in vec4 rgba, const HIGHP in vec2 st, const HIGHP in float time) {
    #ifdef DITHER_TRIANGLENOISE_CHROMATIC 
    vec3 dither = vec3(
            triangleNoise(st, time),
            triangleNoise(st + 0.1337, time),
            triangleNoise(st + 0.3141, time));
    #else
    vec3 dither = vec3(triangleNoise(st, time));
    #endif
    return (rgba + vec4(dither, dither.x)) / 255.0;
}

#if defined(RESOLUTION)
float ditherTriangleNoise(const in float b, const HIGHP in float time) {
    return ditherTriangleNoise(b, gl_FragCoord.xy / RESOLUTION, time);
}

vec3 ditherTriangleNoise(const in vec3 rgb, const HIGHP in float time) {
    return ditherTriangleNoise(rgb, gl_FragCoord.xy / RESOLUTION, time);
}

vec4 ditherTriangleNoise(const in vec4 b, const HIGHP in float time) {
    return ditherTriangleNoise(b, gl_FragCoord.xy / RESOLUTION, time);
}
#endif

#endif