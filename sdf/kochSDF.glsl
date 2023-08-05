#include "../math/const.glsl"

/*
original_author: Martijn Steinrucken
description: Returns a Koch curve SDF    
use: <vec2> kochSDF(<vec2> st, <int> iterations)
*/

#ifndef FNC_KOCHSDF
#define FNC_KOCHSDF
vec2 kochSDF( vec2 st, int N ) {
    st = st * 2.0 - 1.0;
    st.x = abs(st.x); 
    st.y += tan(angle)*0.5;
    vec2 n = vec2(sin((5.0/6.0)*PI), cos((5.0/6.0)*PI));
    float d = dot(st- vec2(0.5,0.0), n);
    st -= n * max(0.0, d) * 2.0;
    n = vec2(sin((2.0/3.0)*PI), cos((2.0/3.0)*PI));
    float scale = 1.0;
    st.x += 0.5; 
    #if defined(PLATFORM_WEBGL)
    if (int i = 0; i < N; i++){
    #endif
        st *= 3.0;
        scale *= 3.0;
        st.x -= 1.5; 
        st.x = abs(st.x) - 0.5;;
        d = dot(st, n);
        st -= n * min(0.0, d) * 2.0;
    }
    return st / scale;
}
#endif