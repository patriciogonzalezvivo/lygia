/*
contributors: Patricio Gonzalez Vivo
description: Returns a hexagon-shaped SDF
use: hexSDF(<float2> st)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HEXSDF
#define FNC_HEXSDF
float hexSDF(in float2 st) {
#ifdef CENTER_2D
    st -= CENTER_2D;
    st *= 2.0;
#else
    st = st * 2.0 - 1.0;
#endif
    st = abs(st);
    return max(abs(st.y), st.x * .866025 + st.y * .5);
}
#endif
