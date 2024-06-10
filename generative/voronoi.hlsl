
#include "../math/const.hlsl"
#include "random.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Voronoi positions and distance to centroids
use: <float3> voronoi(<float2> pos, <float> time)
options:
  VORONOI_RANDOM_FNC: null
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef VORONOI_RANDOM_FNC 
#define VORONOI_RANDOM_FNC(UV) ( 0.5 + 0.5 * sin(time + TAU * random2(UV) ) ); 
#endif

#ifndef FNC_VORONOI
#define FNC_VORONOI
float3 voronoi(float2 uv, float time) {
    float2 i_uv = floor(uv);
    float2 f_uv = frac(uv);
    float3 rta = float3(0.0, 0.0, 10.0);
    for (int j=-1; j<=1; j++ ) {
        for (int i=-1; i<=1; i++ ) {
            float2 neighbor = float2(float(i),float(j));
            float2 p = VORONOI_RANDOM_FNC(i_uv + neighbor);
            p = 0.5 + 0.5 * sin(time + TAU * p);
            float2 diff = neighbor + p - f_uv;
            float dist = length(diff);
            if ( dist < rta.z ) {
                rta.xy = p;
                rta.z = dist;
            }
        }
    }
    return rta;
}

float3 voronoi(float2 p)  { return voronoi(p, 0.0); }
float3 voronoi(float3 p)  { return voronoi(p.xy, p.z); }
#endif
