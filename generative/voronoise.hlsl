
#include "random.hlsl"
#include "../math/const.hlsl"

/*
contributors: Inigo Quiles
description: Cell noise https://iquilezles.org/articles/voronoise/
use: <float> voronoi(<float3|float2> pos, float voronoi, float _smoothness);
options:
    VORONOI_RANDOM_FNC: 
*/

#ifndef VORONOI_RANDOM_FNC 
#define VORONOI_RANDOM_FNC(XYZ) random3(XYZ) 
#endif

#ifndef FNC_VORONOISE
#define FNC_VORONOISE
float voronoise( in float2 p, in float u, float v) {
    float k = 1.0+63.0*pow(1.0-v,6.0);
    float2 i = floor(p);
    float2 f = frac(p);
    
    float2 a = float2(0.0, 0.0);
    float2 g = float2(-2.0, -2.0);
    for( g.y = -2.0; g.y <= 2.0; g.y++ )
    for( g.x = -2.0; g.x <= 2.0 ; g.x++ ) {
        float3  o = VORONOI_RANDOM_FNC(i + g) * float3(u, u, 1.0);
        float2  d = g - f + o.xy;
        float   w = pow(1.0-smoothstep(0.0,1.414, length(d)), k );
        a += float2(o.z*w,w);
    }
        
    return a.x/a.y;
}

float voronoise(float3 p, float u, float v)  {
    float k = 1.0+63.0*pow(1.0-v,6.0);
    float3 i = floor(p);
    float3 f = frac(p);

    float s = 1.0 + 31.0 * v;
    float2 a = float2(0.0, 0.0);
    float3 g = float3(-2.0, -2.0, -2.0);
    for (g.z=-2.; g.z <= 2.0; g.z++)
    for (g.y=-2.; g.y <= 2.0; g.y++)
    for (g.x=-2.; g.x <= 2.0; g.x++) {
        float3 o = VORONOI_RANDOM_FNC(i + g) * float3(u, u, 1.);
        float3 d = g - f + o + 0.5;
        float w = pow(1.0 - smoothstep(0.0, 1.414, length(d)), k);
        a += float2(o.z*w, w);
     }
     return a.x / a.y;
}

#endif
