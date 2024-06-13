#include "../../math/decimate.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: "2016, Banding in Games: A Noisy Rant"
use:
    - <vec4|vec3|float> ditherTriangleNoise(<vec4|vec3|float> value, <vec2> st, <float> time)
    - <vec4|vec3|float> ditherTriangleNoise(<vec4|vec3|float> value, <float> time)
examples:
    - /shaders/color_dither.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef DITHER_TRIANGLENOISE_COORD
#define DITHER_TRIANGLENOISE_COORD gl_FragCoord.xy
#endif

#ifdef DITHER_TIME
#define DITHER_TRIANGLENOISE_TIME DITHER_TIME
#endif

#ifdef DITHER_CHROMATIC
#define DITHER_TRIANGLENOISE_CHROMATIC
#endif

#ifndef DITHER_TRIANGLENOISE_PRECISION
#ifdef DITHER_PRECISION
#define DITHER_TRIANGLENOISE_PRECISION DITHER_PRECISION
#else
#define DITHER_TRIANGLENOISE_PRECISION 255
#endif
#endif

#ifndef FNC_DITHER_TRIANGLENOISE
#define FNC_DITHER_TRIANGLENOISE

float triangleNoise(HIGHP in vec2 st) {
    st = floor(st);
    #ifdef DITHER_TRIANGLENOISE_TIME
    st += vec2(0.07 * fract(DITHER_TRIANGLENOISE_TIME));
    #endif
    st  = fract(st * vec2(5.3987, 5.4421));
    st += dot(st.yx, st.xy + vec2(21.5351, 14.3137));

    HIGHP float xy = st.x * st.y;
    return (fract(xy * 95.4307) + fract(xy * 75.04961) - 1.0);
}

vec3 ditherTriangleNoise(const in vec3 color, const HIGHP in vec2 st, const int pres) {
    
    #ifdef DITHER_TRIANGLENOISE_CHROMATIC 
    vec3 ditherPattern = vec3(
            triangleNoise(st),
            triangleNoise(st + 0.1337),
            triangleNoise(st + 0.3141));
    #else
    vec3 ditherPattern = vec3(triangleNoise(st));
    #endif
    
    // return color + ditherPattern / 255.0;
    float d = float(pres);
    float h = 0.5/d;
    return decimate(color - h + ditherPattern / d, d);
}

float ditherTriangleNoise(const in float b, const HIGHP in vec2 st) { return b + triangleNoise(st) / float(DITHER_TRIANGLENOISE_PRECISION); }
vec3 ditherTriangleNoise(const in vec3 color, const in vec2 xy) {  return ditherTriangleNoise(color, xy, DITHER_TRIANGLENOISE_PRECISION); }
vec4 ditherTriangleNoise(const in vec4 color, const in vec2 xy) {  return vec4(ditherTriangleNoise(color.rgb, xy, DITHER_TRIANGLENOISE_PRECISION), color.a); }

float ditherTriangleNoise(const in float val, int pres) { return ditherTriangleNoise(vec3(val),DITHER_TRIANGLENOISE_COORD, pres).r; }
vec3 ditherTriangleNoise(const in vec3 color, int pres) { return ditherTriangleNoise(color, DITHER_TRIANGLENOISE_COORD, pres); }
vec4 ditherTriangleNoise(const in vec4 color, int pres) { return vec4(ditherTriangleNoise(color.rgb, DITHER_TRIANGLENOISE_COORD, pres), color.a); }

float ditherTriangleNoise(const in float val) { return ditherTriangleNoise(vec3(val), DITHER_TRIANGLENOISE_COORD, DITHER_TRIANGLENOISE_PRECISION).r; }
vec3 ditherTriangleNoise(const in vec3 color) { return ditherTriangleNoise(color, DITHER_TRIANGLENOISE_COORD, DITHER_TRIANGLENOISE_PRECISION); }
vec4 ditherTriangleNoise(const in vec4 color) { return vec4(ditherTriangleNoise(color.rgb), color.a); }
#endif