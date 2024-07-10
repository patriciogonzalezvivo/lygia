#include "nyquist.glsl"

/*
contributors: dahart (https://www.shadertoy.com/user/dahart)
description: |
    Anti-aliasing fract function. It clamp except for a 2-pixel wide gradient along the edge
    Based on this example https://www.shadertoy.com/view/4l2BRD
use: <float> aafract(<float> x)
option:
    AA_EDGE: in the absence of derivatives you can specify the antialiasing factor
*/

#ifndef FNC_AAFRACT
#define FNC_AAFRACT

#if defined(GL_OES_standard_derivatives)
#extension GL_OES_standard_derivatives : enable
#endif

float aafract(float x) {
#if !defined(GL_ES) || __VERSION__ >= 300 || defined(GL_OES_standard_derivatives)
    float afwidth = 2.0 * length(vec2(dFdx(x), dFdy(x)));
    float fx = fract(x);
    float idx = 1. - afwidth;
    float v = (fx < idx) ? fx/idx : (1.0-fx)/afwidth;
    return nyquist(v, afwidth);
#elif defined(AA_EDGE)
    float afwidth = AA_EDGE;
    float fx = fract(x);
    float idx = 1. - afwidth;
    float v = (fx < idx) ? fx/idx : (1.0-fx)/afwidth;
    return nyquist(v, afwidth);
#else 
    return fract(x);
#endif
}

vec2 aafract(vec2 v) { return vec2(aafract(v.x), aafract(v.y)); }

#endif