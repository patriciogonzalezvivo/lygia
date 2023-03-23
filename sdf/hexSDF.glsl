/*
original_author: Patricio Gonzalez Vivo
description: Returns a hexagon-shaped SDF
use: hexSDF(<vec2> st)
*/

#ifndef FNC_HEXSDF
#define FNC_HEXSDF
float hexSDF(in vec2 st) {
    st = abs(st * 2. - 1.);
    return max(abs(st.y), st.x * .866025 + st.y * .5);
}
#endif
