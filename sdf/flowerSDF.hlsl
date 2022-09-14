/*
original_author: Patricio Gonzalez Vivo
description: Returns a flower shaped SDF
use: flowerSDF(<float2> st, <int> n_sides)
*/

#ifndef FNC_FLOWERSDF
#define FNC_FLOWERSDF
float flowerSDF(float2 st, int N) {
    st = st * 2.0 - 1.0;
    float r = length(st) * 2.0;
    float a = atan2(st.y, st.x);
    float v = float(N) * 0.5;
    return 1.0 - (abs(cos(a * v)) *  0.5 + 0.5) / r;
}
#endif