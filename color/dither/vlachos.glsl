/*
contributors: Patricio Gonzalez Vivo
description: |
    Vlachos 2016, "Advanced VR Rendering" http://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
use: <vec4|vec3|float> ditherVlachos(<vec4|vec3|float> value, <float> time)
options:
    - DITHER_VLACHOS_TIME
    - DITHER_VLACHOS_CHROMATIC
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

#ifdef DITHER_TIME
#define DITHER_VLACHOS_TIME DITHER_TIME
#endif

#ifndef DITHER_VLACHOS_COORD
#define DITHER_VLACHOS_COORD gl_FragCoord.xy
#endif

#ifndef FNC_DITHER_VLACHOS
#define FNC_DITHER_VLACHOS

float ditherVlachos(float b) {
    vec2 st = DITHER_VLACHOS_COORD;
    #ifdef DITHER_VLACHOS_TIME
    st += 1337.0*fract(DITHER_VLACHOS_TIME);
    #endif
    HIGHP float noise = dot(vec2(171.0, 231.0), st);
    noise = fract(noise / 71.0);
    noise = (noise * 2.0) - 1.0;
    return b + (noise / 255.0);
}

vec3 ditherVlachos(vec3 color) {
    vec2 st = DITHER_VLACHOS_COORD;
    #ifdef DITHER_VLACHOS_TIME
    st += 1337.0*fract(DITHER_VLACHOS_TIME);
    #endif
    HIGHP vec3 noise = vec3(dot(vec2(171.0, 231.0), st));
    noise = fract(noise / vec3(103.0, 71.0, 97.0));
    return color.rgb + (noise / 255.0);
}

vec4 ditherVlachos(vec4 color) {
    return vec4(ditherVlachos(color.rgb), color.a);
}

#endif