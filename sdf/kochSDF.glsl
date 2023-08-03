#include "../math/const.glsl"

/*
original_author: The Art of Code
description: Returns a Koch curve SDF    
use: <vec2> kochSDF(<vec2> st, <int> iterations)
*/

#ifndef FNC_KOCHSDF
#define FNC_KOCHSDF
vec2 kochSDF( vec2 st, int N ) {
    float angle = (5.0/6.0)*PI;
    st = st * 2.0 - 1.0;
    st.x = abs(st.x); 
    st.y += tan(angle)*0.5;
    vec2 n = vec2(sin(angle), cos(angle));
    float d = dot(st- vec2(0.5,0.0), n);
    st -= n * max(0.0, d) * 2.0;
    n = vec2(sin((2.0/3.0)*PI), cos((2.0/3.0)*PI));
    float scale = 1.0;
    st.x += 0.5; 
    if (N == 1){
        st *= 3.0;
        scale *= 3.0;
        st.x -= 1.5; 
        st.x = abs(st.x) - 0.5;;
        d = dot(st, n);
        st -= n * min(0.0, d) * 2.0;
    } else if (N == 2){
        for (int i = 0; i < 2; i++) {
            st *= 3.0;
            scale *= 3.0;
            st.x -= 1.5; 
            st.x = abs(st.x) - 0.5;;
            d = dot(st, n);
            st -= n * min(0.0, d) * 2.0;
       }
    } else if (N == 3){
        for (int i = 0; i < 3; i++) {
            st *= 3.0;
            scale *= 3.0;
            st.x -= 1.5; 
            st.x = abs(st.x) - 0.5;;
            d = dot(st, n);
            st -= n * min(0.0, d) * 2.0;
        }
    }
    return st / scale;
}
#endif