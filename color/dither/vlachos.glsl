/*
original_author: Patricio Gonzalez Vivo
description: |
    Vlachos 2016, "Advanced VR Rendering" http://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
use: <vec4|vec3|float> ditherVlachos(<vec4|vec3|float> value, <float> time)
options:
    - DITHER_VLACHOS_ANIMATED
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

#ifdef DITHER_ANIMATED
#define DITHER_VLACHOS_ANIMATED
#endif

#ifndef DITHER_VLACHOS
#define DITHER_VLACHOS

float ditherVlachos(float b, const HIGHP in float time) {
    vec2 st = gl_FragCoord.xy;
    #ifdef DITHER_VLACHOS_ANIMATED
    st += 1337.0*fract(time);
    #endif
    HIGHP float noise = dot(vec2(171.0, 231.0), st);
    noise = fract(noise / 71.0);
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return b + (noise / 255.0);
}

vec3 ditherVlachos(vec3 rgb, const HIGHP in float time) {
    vec2 st = gl_FragCoord.xy;
    #ifdef DITHER_VLACHOS_ANIMATED
    st += 1337.0*fract(time);
    #endif
    HIGHP vec3 noise = vec3(dot(vec2(171.0, 231.0), st));
    noise = fract(noise / vec3(103.0, 71.0, 97.0));
    return rgb.rgb + (noise / 255.0);
}

vec4 ditherVlachos(vec4 rgba, const HIGHP in float time) {
    return vec4(ditherVlachos(rgba.rgb, time), rgba.a);
}

#endif