/*
contributors: Matt DesLauriers
description: Performs a smoothstep using standard derivatives for anti-aliased edges at any level of magnification. From https://github.com/glslify/glsl-aastep
use: aastep(<float> threshold, <float> value)
option:
    AA_EDGE: in the absence of derivatives you can specify the antialiasing factor
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/draw/aastep.frag
*/

#ifndef FNC_AASTEP
#define FNC_AASTEP

#if defined(GL_OES_standard_derivatives)
#extension GL_OES_standard_derivatives : enable
#endif

float aastep(float threshold, float value) {
#if !defined(GL_ES) || __VERSION__ >= 300 || defined(GL_OES_standard_derivatives)
    float afwidth = 0.7 * length(vec2(dFdx(value), dFdy(value)));
    return smoothstep(threshold-afwidth, threshold+afwidth, value);
#elif defined(AA_EDGE)
    float afwidth = AA_EDGE;
    return smoothstep(threshold-afwidth, threshold+afwidth, value);
#else 
    return step(threshold, value);
#endif
}
#endif