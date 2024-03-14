#include "map.hlsl"

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
float aafloor(float x) {
#if defined(AA_EDGE)
    float afwidth = AA_EDGE;
#else 
    float afwidth = 2.0 * fwidth(x);
#endif
    float fx = frac(x);
    float idx = 1. - afwidth;
    return (fx < idx) ? x - fx : map(fx, idx, 1., x-fx, x);
}

float2 aafloor(float2 x) { return float2(aafloor(x.x), aafloor(x.y)); }
#endif