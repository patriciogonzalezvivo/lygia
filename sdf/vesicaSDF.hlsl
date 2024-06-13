#include "circleSDF.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns an almond-shaped sdf
use: vesicaSDF(<float2> st, <float> w)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_VESICASDF
#define FNC_VESICASDF
float vesicaSDF(in float2 st, in float w) {
    float2 offset = float2(w*0.5,0.);
    return max( circleSDF(st-offset),
                circleSDF(st+offset));
}

float vesicaSDF(in float2 st) {
    return vesicaSDF(st, 0.5);
}
#endif
