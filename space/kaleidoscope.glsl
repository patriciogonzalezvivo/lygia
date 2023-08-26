/*
original_author:  Martijn Steinrucken
description: It mirrors uvs in a kaleisdoscope pattern.  Based on [Koch Snowflake tutorial](https://www.shadertoy.com/view/tdcGDj) 
use: <vec2> kaleidoscope(<vec2> st)
*/

#ifndef FNC_KALEIDOSCOPE
#define FNC_KALEIDOSCOPE
vec2 kaleidoscope(vec2 st){
    #ifdef CENTER_2D
    st -= CENTER_2D;
    #else
    st -= vec2(0.5);
    #endif
    float r3 = sqrt(3.);
    st = abs(st);  
    vec2 dir = vec2(1.0, -r3)*0.5;
    float d = dot(st, dir);  
    st -= dir * max(0.0, d) * 2.0; 
    dir = vec2(r3, -1.0)*0.5;
    st -= dir * min(0., dot(st, dir)) * 2.0;
    return st;
}
#endif