#include "random.hlsl"
#include "../math/distance.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Worley noise
use: <float2> worley(<float2|float3> pos)
options:
    - DIST_FNC: change the distance function, currently implemented are euclidean, manhattan, chebychev and minkowski
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_WORLEY
#define FNC_WORLEY

#ifndef DIST_FNC
#define DIST_FNC euclidean
#endif

float worley(float2 p){
    float2 n = floor( p );
    float2 f = frac( p );

    float dis = 1.0;
    for( int j= -1; j <= 1; j++ )
        for( int i=-1; i <= 1; i++ ) {	
                float2  g = float2(i,j);
                float2  o = random2( n + g );
                float d = DIST_FNC(g+o, f);
                dis = min(dis,d);
    }

    return 1.0-dis;
}

float worley(float3 p){
    float3 n = floor( p );
    float3 f = frac( p );

    float dis = 1.0;
    for( int k = -1; k <= 1; k++ )
        for( int j= -1; j <= 1; j++ )
            for( int i=-1; i <= 1; i++ ) {	
                float3  g = float3(i,j,k);
                float3  o = random3( n + g );
                float d = DIST_FNC(g+o, f);
                dis = min(dis,d);
    }

    return 1.0-dis;
}

#endif