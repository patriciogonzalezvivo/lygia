/*
original_author:  Martijn Steinrucken
description: It mirrors uvs in a kaleisdoscope pattern.  Based on [Koch Snowflake tutorial](https://www.shadertoy.com/view/tdcGDj) 
use: <float2> kaleidoscope(<float2> st)
*/

#ifndef FNC_KALEIDOSCOPE
#define FNC_KALEIDOSCOPE
float2 kaleidoscope(float2 st){
    #ifdef CENTER_2D
    st -= CENTER_2D;
    #else
    st -= float2(0.5, 0.5);
    #endif
    float r3 = sqrt(3.);
    st = abs(st);  
    float2 dir = float2(1.0, -r3)*0.5;
    float d = dot(st, dir);  
    st -= dir * max(0.0, d) * 2.0; 
    dir = float2(r3, -1.0)*0.5;
    st -= dir * min(0., dot(st, dir)) * 2.0;
    return st;
}
#endif