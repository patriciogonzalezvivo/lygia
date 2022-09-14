#include "triSDF.hlsl"

/*
description: Returns a rhomb-shaped sdf
use: rhombSDF(<float2> st)
original_author: Patricio Gonzalez Vivo
*/

#ifndef FNC_RHOMBSDF
#define FNC_RHOMBSDF
float rhombSDF(float2 st) {
    return max(triSDF(st),
               triSDF(float2(st.x, 1. - st.y)));
}
#endif
