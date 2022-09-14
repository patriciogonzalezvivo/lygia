#include "circleSDF.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: Returns an almond-shaped sdf
use: vesicaSDF(<float2> st, <float> w)
*/

#ifndef FNC_VESICASDF
#define FNC_VESICASDF
float vesicaSDF(in float2 st, in float w) {
    float2 offset = float2(w*.5,0.);
    return max( circleSDF(st-offset),
                circleSDF(st+offset));
}
#endif
