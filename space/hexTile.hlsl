#include "scale.hlsl"
#include "../math/mod.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: make some hexagonal tiles. XY provide coordenates of the tile. While Z provides the distance to the center of the tile
use: <float3> hexTile(<float2> st [, <float> scale])
options:
    HEXTILE_SIZE
    HEXTILE_H
    HEXTILE_S
*/

#ifndef FNC_HEXTILE
#define FNC_HEXTILE
float3 hexTile(in float2 st, in float scl) {
    // this is hack to scale the hexagon to be more squared
    st = scale(st, float2(1.24,1.)*scl) + float2(.5,0.);

    float3 q = float3(st, .0);
    q.z = -.5 * q.x - q.y;

    float z = -.5 * q.x - q.y;
    q.y -= .5 * q.x;

    float3 i = floor(q+.5);
    float s = floor(i.x + i.y + i.z);
    float3 d = abs(i-q);

    // TODO: all this ifs should be avoided
    if( d.x >= d.y && d.x >= d.z ) i.x -= s;
    else if( d.y >= d.x && d.y >= d.z )	i.y -= s;
    else i.z -= s;

    float2 coord = float2(i.x, (i.y - i.z + (1.-mod(i.x, 2.)))/2.);
    float dist = length(st - float2(coord.x, coord.y - .5*mod(i.x-1., 2.)));
    return float3(coord, dist);
}

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