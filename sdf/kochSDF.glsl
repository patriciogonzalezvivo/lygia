#include "../math/const.glsl"

/*
contributors: Kathy kfahn22
description: Returns a Koch curve SDF. 
use: <vec2> kochSDF(<vec2> st, <int> iterations)
*/

#ifndef FNC_KOCHSDF
#define FNC_KOCHSDF
float kochSDF( vec2 st, vec2 center, int N ) {
    st -= center;
    st *= 3.0;
    float r3 = sqrt(3.);  
    st = abs(st);
    st += r3*vec2(-st.y,st.x); // 60° rotation, scale 2
    st.y -= 1.;   
    float w = .5;    
    mat2 m = mat2(r3,3,-3,r3)*.5;
    #ifdef PLATFORM_WEBGL
    for (int i = 0; i< 20; i++) {
        if (i >= N) break;
    #else
    for (int i = 0; i< N; i++) {
    #endif
        st = vec2(-r3,3)*.5 - m*vec2(st.y,abs(st.x));
        w /= r3;
    }
    float d = sign(st.y)*length(vec2(st.y,max(0.,abs(st.x)-r3)));  
    return (d*w);
}

float kochSDF( vec2 st, int N ) {
    #ifdef CENTER_2D
        return kochSDF(st, CENTER_2D, N);
    #else
        return kochSDF(st, vec2(0.5), N);
    #endif
}
#endif