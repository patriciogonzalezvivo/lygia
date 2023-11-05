/*
contributors: Patricio Gonzalez Vivo
description: |
    Fix the aspect ratio of a space keeping things squared for you, 
    in a similar way that aspect.glsl does, but while scaling the 
    space to keep the entire 0.0,0.0 ~ 1.0,1.0 range visible
    
use: <vec2> ratio(<vec2> st, <vec2> st_size)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_RATIO
#define FNC_RATIO
vec2 ratio(in vec2 v, in vec2 s) {
    return mix( vec2((v.x*s.x/s.y)-(s.x*.5-s.y*.5)/s.y,v.y),
                vec2(v.x,v.y*(s.y/s.x)-(s.y*.5-s.x*.5)/s.x),
                step(s.x,s.y));
}
#endif
