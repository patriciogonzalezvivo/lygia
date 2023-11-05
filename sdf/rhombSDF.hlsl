#include "triSDF.hlsl"

/*
description: Returns a rhomb-shaped sdf
use: rhombSDF(<float2> st)
contributors: Patricio Gonzalez Vivo
*/

#ifndef FNC_RHOMBSDF
#define FNC_RHOMBSDF
float rhombSDF(in float2 st) {
    float offset = 1.0;
    #ifdef CENTER_2D
    offset = CENTER_2D.y * 2.0;
    #endif 
    return max(triSDF(st),
               triSDF(float2(st.x, offset-st.y)));

}
#endif
