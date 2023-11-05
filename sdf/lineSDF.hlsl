/*
contributors: Inigo Quiles
description: Segment SDF
use: lineSDF(<float2> st, <float2> A, <float2> B)
*/

#ifndef FNC_LINESDF
#define FNC_LINESDF
float lineSDF( in float2 st, in float2 a, in float2 b ) {
    float2 b_to_a = b - a;
    float2 to_a = st - a;
    float h = saturate(dot(to_a, b_to_a)/dot(b_to_a, b_to_a));
    return length(to_a - h * b_to_a );
}
#endif
