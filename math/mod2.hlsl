#include "mod.hlsl"

/*
contributors: Mercury
description: |
    Two dimensional modulus, returns the remainder of a division of two vectors.
    Found at in Mercury's library https://mercury.sexy/hg_sdf/
use: <float2> mod2(<float2> x, <float2> size)
*/

#ifndef FNC_MOD2
#define FNC_MOD2
float2 mod2(inout float2 p, float s) {
    float2 c = floor((p + s * 0.5) / s);
    p = mod(p + s*0.5,s) - s*0.5;
    return c;
}

float2 mod2(inout float2 p, float2 s) {
    float2 c = floor((p + s * 0.5) / s);
    p = mod(p + s*0.5,s) - s*0.5;
    return c;
}
#endif
