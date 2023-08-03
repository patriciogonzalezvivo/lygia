/*
original_author: VHARIVINAY
description: |
    Returns the Juia set SDF
    For more information about the Julia set, check [this article](https://en.wikipedia.org/wiki/Julia_set)
    Some values for c:
        * vec2(−0.8, 0.156)
        * vec2(0.285, 0.0)
        * vec2(−0.7269, 0.1889)
        * vec2(0.27334, 0.00742)
        * vec2(−0.835, −0.2321)
use: juliaSDF(<float2> st, <float2> c, <float> r)
*/

#ifndef FNC_JULIASDF
#define FNC_JULIASDF
float juliaSDF( float2 st, float2 c, float r) {
    st = st * 2.0 - 1.0;
    float2 z = float2(0.0) - (st) * r;
    float n = 0.0;
    const int I = 500;
    for (int i = I; i > 0; i --) { 
        if ( z.x*z.x + z.y*z.y > 4.0 ) { 
        n = float(i)/float(I); 
        break;
        } 
        z = float2( (z.x*z.x - z.y*z.y) + c.x, (2.0*z.x*z.y) + c.y ); 
    } 
    return n;
}
#endif