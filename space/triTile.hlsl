/*
contributors: Mathias Bredholt
description: |
    make some triangular tiles. XY provide coords inside of the tile. ZW provides tile coords
use: <float4> triTile(<float2> st [, <float> scale])
*/

#ifndef FNC_TRITILE
#define FNC_TRITILE
float4 triTile(float2 st) {
    st = mult(float2x2( 1.0, -1.0 / 1.7320508, 
                        0.0, 2.0 / 1.7320508), st);
    float4 f = float4(st, -st);
    float4 i = floor(f);
    f = frac(f);
    return dot(f.xy, f.xy) < dot(f.zw, f.zw)
                ? float4(f.xy, float2(2., 1.) * i.xy)
                : float4(f.zw, -(float2(2., 1.) * i.zw + 1.));
}

float4 triTile(float2 st, float scale) { return triTile(st * scale); }
#endif
