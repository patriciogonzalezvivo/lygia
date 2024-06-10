#include "triSDF.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns a rhomb-shaped sdf
use: rhombSDF(<float2> st)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
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
