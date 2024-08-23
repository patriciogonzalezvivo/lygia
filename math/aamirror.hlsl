#include "nyquist.hlsl"

/*
contributors: Shadi El Hajj
description: An anti-aliased triangle wave function.
use: <float> mirror(<float> x)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_AAMIRROR
#define FNC_AAMIRROR

float aamirror(float x) {
#if defined(AA_EDGE)
    float afwidth = AA_EDGE;
#else 
    float afwidth = length(float2(ddx(x),ddy(x)));
#endif
    float v = abs(x - floor(x + 0.5)) * 2.0;
    return nyquist(v, afwidth);
}

#endif

