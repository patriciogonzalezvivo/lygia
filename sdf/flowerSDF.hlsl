/*
contributors: Patricio Gonzalez Vivo
description: Returns a flower shaped SDF
use: flowerSDF(<float2> st, <int> n_sides)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_FLOWERSDF
#define FNC_FLOWERSDF
float flowerSDF(float2 st, int N) {
#ifdef CENTER_2D
    st -= CENTER_2D;
#else
    st -= 0.5;
#endif
    st *= 4.0;
    float r = length(st) * 2.0;
    float a = atan2(st.y, st.x);
    float v = float(N) * 0.5;
    return 1.0 - (abs(cos(a * v)) *  0.5 + 0.5) / r;
}
#endif