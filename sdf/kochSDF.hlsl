#include "../math/const.hlsl"

/*
contributors: Martijn Steinrucken
description: Returns a Koch curve SDF   
use: <float2> kochSDF(<float2> st, <int> iterations)
*/

#ifndef FNC_KOCHSDF
#define FNC_KOCHSDF
float kochSDF( float2 st, float2 center, int N ) {
    st -= center;
    st *= 3.0;
    float r3 = sqrt(3.);  
    st = abs(st);
    st += r3*float2(-st.y,st.x); // 60° rotation, scale 2
    st.y -= 1.;   
    float w = .5;    
    float2x2 m = float2x2(r3,3,-3,r3)*.5;
    for (int i = 0; i< N; i++) {
        st = float2(-r3,3)*.5 - m*float2(st.y,abs(st.x));
        w /= r3;
    }
    float d = sign(st.y)*length(float2(st.y,max(0.,abs(st.x)-r3)));  
    return (d*w);
}

float kochSDF( float2 st, int N ) {
    #ifdef CENTER_2D
        return kochSDF(st, CENTER_2D, N);
    #else
        return kochSDF(st, float2(0.5, 0.5), N);
    #endif
}
#endif