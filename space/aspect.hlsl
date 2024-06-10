/*
contributors: Patricio Gonzalez Vivo
description: 'Fix the aspect ratio of a space keeping things squared for you.'
use: <float2> aspect(<float2> st, <float2> st_size)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ASPECT
#define FNC_ASPECT

float2 aspect(float2 st, float2 s) {
    st.x = st.x * (s.x / s.y);
    return st;
}

#endif