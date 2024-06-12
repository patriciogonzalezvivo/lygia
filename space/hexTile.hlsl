/*
contributors: Patricio Gonzalez Vivo
description: make some hexagonal tiles. XY provide coordenates of the tile. While Z provides the distance to the center of the tile
use: <float4> hexTile(<float2> st)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HEXTILE
#define FNC_HEXTILE

float4 hexTile(float2 st) {
    float2 s = float2(1., 1.7320508);
    float2 o = float2(.5, 1.);
    st = st.yx;
    
    float4 i = floor(float4(st,st-o)/s.xyxy)+.5;
    float4 f = float4(st-i.xy*s, st-(i.zw+.5)*s);
    
    return dot(f.xy,f.xy) < dot(f.zw,f.zw) ? 
            float4(f.yx+.5, i.xy):
            float4(f.wz+.5, i.zw+o);
}

#endif