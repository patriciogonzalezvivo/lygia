/*
contributors: Patricio Gonzalez Vivo
description: |
    Fix the aspect ratio of a space keeping things squared for you, 
    in a similar way that aspect.glsl does, but while scaling the 
    space to keep the entire 0.0,0.0 ~ 1.0,1.0 range visible
use: <float2> ratio(<float2> st, <float2> st_size)
*/

#ifndef FNC_RATIO
#define FNC_RATIO
float2 ratio(in float2 st, in float2 s) {
    return lerp(    float2((st.x*s.x/s.y)-(s.x*.5-s.y*.5)/s.y,st.y),
                    float2(st.x,st.y*(s.y/s.x)-(s.y*.5-s.x*.5)/s.x),
                    step(s.x, s.y));
}
#endif
