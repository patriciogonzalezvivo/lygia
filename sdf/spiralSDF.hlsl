/*
contributors: Patricio Gonzalez Vivo
description: Returns a spiral SDF
use: spiralSDF(<float2> st, <float> turns)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SPIRALSDF
#define FNC_SPIRALSDF
float spiralSDF(float2 st, float t) {
#ifdef CENTER_2D
    st -= CENTER_2D;
#else
    st -= 0.5;
#endif
    float r = dot(st, st);
    float a = atan2(st.y, st.x);
    return abs(sin(frac(log(r) * t + a * 0.159)));
}
#endif