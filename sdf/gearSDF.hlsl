#include "../math/map.hlsl"

/*
original_author: Kathy McGuiness
description: |
    Returns a gear shaped SDF
    Some notes about the parameters:
        * b determines the length and roundness of the spokes
        * n is the number of spokes 
use: gearSDF(<float2> st, <float> b, <int> n_spokes)
*/

#ifndef FNC_GEARSDF
#define FNC_GEARSDF
float hyperbolicTan( float theta ) {
    float e = 2.71828182845904523536;
    float l = pow(e, 2.0 * theta);
    return (l - 1.0) / (l + 1.0);
}
float gearSDF( float2 st, float b, int N ) {
    st = st * 2.0 - 1.0;
    float s = map(b, 1.0, 15.0, 0.066, 0.5);
    float d = length(st) - s;
    float theta = atan(st.y, st.x);
    float r = (1.0/b)*(hyperbolicTan(b * sin(float(N)*theta)));
    return d + min(d, r);
}
#endif