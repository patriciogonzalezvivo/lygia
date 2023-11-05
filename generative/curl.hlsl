#include "snoise.hlsl"

/*
contributors: Isaac Cohen
description: https://github.com/cabbibo/glsl-curl-noise/blob/master/curl.hlsl
use: curl(<float3|float4> pos)
*/


#ifndef FNC_CURL
#define FNC_CURL

float2 curl( float2 p ){
    const float e = .1;
    float2 dx = float2( e   , 0.0 );
    float2 dy = float2( 0.0 , e   );

    float2 p_x0 = snoise2( p - dx );
    float2 p_x1 = snoise2( p + dx );
    float2 p_y0 = snoise2( p - dy );
    float2 p_y1 = snoise2( p + dy );

    float x = p_x1.y + p_x0.y;
    float y = p_y1.x - p_y0.x;

    const float divisor = 1.0 / ( 2.0 * e );
    return normalize( float2(x , y) * divisor );
}

float3 curl( float3 p ){
    const float e = .1;
    float3 dx = float3( e   , 0.0 , 0.0 );
    float3 dy = float3( 0.0 , e   , 0.0 );
    float3 dz = float3( 0.0 , 0.0 , e   );

    float3 p_x0 = snoise3( p - dx );
    float3 p_x1 = snoise3( p + dx );
    float3 p_y0 = snoise3( p - dy );
    float3 p_y1 = snoise3( p + dy );
    float3 p_z0 = snoise3( p - dz );
    float3 p_z1 = snoise3( p + dz );

    float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
    float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
    float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

    const float divisor = 1.0 / ( 2.0 * e );
    return normalize( float3( x , y , z ) * divisor );
}

float3 curl( float4 p ){
    const float e = .1;
    float4 dx = float4( e   , 0.0 , 0.0 , 1.0 );
    float4 dy = float4( 0.0 , e   , 0.0 , 1.0 );
    float4 dz = float4( 0.0 , 0.0 , e   , 1.0 );

    float3 p_x0 = snoise3( p - dx );
    float3 p_x1 = snoise3( p + dx );
    float3 p_y0 = snoise3( p - dy );
    float3 p_y1 = snoise3( p + dy );
    float3 p_z0 = snoise3( p - dz );
    float3 p_z1 = snoise3( p + dz );

    float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
    float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
    float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

    const float divisor = 1.0 / ( 2.0 * e );
    return normalize( float3( x , y , z ) * divisor );
}

#endif