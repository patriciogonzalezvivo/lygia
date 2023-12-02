/*
contributors: Patricio Gonzalez Vivo
description: |
    Gjøl 2016, "Banding in Games: A Noisy Rant"
use: 
 - <vec4|vec3|float> ditherTriangleNoise(<vec4|vec3|float> value, <vec2> st, <float> time)
 - <vec4|vec3|float> ditherTriangleNoise(<vec4|vec3|float> value, <float> time)
examples:
    - /shaders/color_dither.frag
*/

#ifndef DITHER_CHROMATIC_COORD
#define DITHER_CHROMATIC_COORD gl_FragCoord.xy
#endif

#ifdef DITHER_TIME
#define DITHER_TRIANGLENOISE_TIME DITHER_TIME
#endif

#ifdef DITHER_CHROMATIC
#define DITHER_TRIANGLENOISE_CHROMATIC
#endif

#ifndef FNC_DITHER_TRIANGLENOISE
#define FNC_DITHER_TRIANGLENOISE

float triangleNoise(HIGHP in vec2 n) {
    // triangle noise, in [-1.0..1.0[ range
    #ifdef DITHER_TRIANGLENOISE_TIME
    n += vec2(0.07 * fract(DITHER_TRIANGLENOISE_TIME));
    #endif
    n  = fract(n * vec2(5.3987, 5.4421));
    n += dot(n.yx, n.xy + vec2(21.5351, 14.3137));

    HIGHP float xy = n.x * n.y;
    // compute in [0..2[ and remap to [-1.0..1.0[
    return fract(xy * 95.4307) + fract(xy * 75.04961) - 1.0;
}

float ditherTriangleNoise(const in float b, const HIGHP in vec2 st) {
    return b + triangleNoise(st) / 255.0;
}

vec3 ditherTriangleNoise(const in vec3 rgb, const HIGHP in vec2 st) {
    
    #ifdef DITHER_TRIANGLENOISE_CHROMATIC 
    vec3 dither = vec3(
            triangleNoise(st),
            triangleNoise(st + 0.1337),
            triangleNoise(st + 0.3141));
    #else
    vec3 dither = vec3(triangleNoise(st));
    #endif
            
    return rgb + dither / 255.0;
}

vec4 ditherTriangleNoise(const in vec4 rgba, const HIGHP in vec2 st) {
    #ifdef DITHER_TRIANGLENOISE_CHROMATIC 
    vec3 dither = vec3(
            triangleNoise(st),
            triangleNoise(st + 0.1337),
            triangleNoise(st + 0.3141));
    #else
    vec3 dither = vec3(triangleNoise(st));
    #endif
    return (rgba + vec4(dither, dither.x)) / 255.0;
}

#if defined(RESOLUTION)
float ditherTriangleNoise(const in float b) {
    return ditherTriangleNoise(b, DITHER_CHROMATIC_COORD / RESOLUTION);
}

vec3 ditherTriangleNoise(const in vec3 rgb) {
    return ditherTriangleNoise(rgb, DITHER_CHROMATIC_COORD / RESOLUTION);
}

vec4 ditherTriangleNoise(const in vec4 b) {
    return ditherTriangleNoise(b, gl_FragCoord.xy / RESOLUTION);
}
#endif

#endif