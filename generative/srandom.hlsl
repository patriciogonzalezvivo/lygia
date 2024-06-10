/*
contributors: Patricio Gonzalez Vivo
description: Signed Random
use: srandomX(<float2|float3> x)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SRANDOM
#define FNC_SRANDOM

float2 srandom2(in float2 st) {
    const float2 k = float2(.3183099, .3678794);
    st = st * k + k.yx;
    return -1. + 2. * frac(16. * k * frac(st.x * st.y * (st.x + st.y)));
}

float3 srandom3(in float3 p) {
    p = float3( dot(p, float3(127.1, 311.7, 74.7)),
            dot(p, float3(269.5, 183.3, 246.1)),
            dot(p, float3(113.5, 271.9, 124.6)));
    return -1. + 2. * frac(sin(p) * 43758.5453123);
}

#endif