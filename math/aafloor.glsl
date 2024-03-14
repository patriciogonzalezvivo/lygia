#include "map.glsl"

/*
contributors: ["dahart", "Fabrice NEYRET"]
description: |
    Is similar to floor() but has a 2-pixel wide gradient between clamped steps 
    to allow the edges in the result to be anti-aliased.
    Based on examples https://www.shadertoy.com/view/4l2BRD and https://www.shadertoy.com/view/3tSGWy
use: <float> aafloor(<float> x)
option:
    AA_EDGE: in the absence of derivatives you can specify the antialiasing factor
*/

#ifndef FNC_AAFLOOR
#define FNC_AAFLOOR

#if defined(GL_OES_standard_derivatives)
#extension GL_OES_standard_derivatives : enable
#endif

float aafloor(float x) {
#if !defined(GL_ES) || __VERSION__ >= 300 || defined(GL_OES_standard_derivatives)
    float afwidth = 2.0 * length(vec2(dFdx(x), dFdy(x)));
    float fx = fract(x);
    float idx = 1. - afwidth;
    return (fx < idx) ? x - fx : map(fx, idx, 1., x-fx, x);
#elif defined(AA_EDGE)
    float afwidth = AA_EDGE;
    float fx = fract(x);
    float idx = 1. - afwidth;
    return (fx < idx) ? x - fx : map(fx, idx, 1., x-fx, x);
#else 
    return floor(x);
#endif
}

vec2 aafloor(vec2 x) { return vec2(aafloor(x.x), aafloor(x.y)); }

#endif