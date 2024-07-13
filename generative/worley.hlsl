#include "random.hlsl"
#include "../math/dist.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Worley noise. Returns float2(F1, F2)
use: <float2> worley2(<float2|float3> pos)
options:
    - WORLEY_JITTER: amount of pattern randomness. With 1.0 being the default and 0.0 resulting in a perfectly symmetrical pattern.
    - WORLEY_DIST_FNC: change the distance function, currently implemented are distEuclidean, distManhattan, distChebychev and distMinkowski
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/generative_worley.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_WORLEY
#define FNC_WORLEY

#ifndef WORLEY_JITTER
#define WORLEY_JITTER 1.0
#endif

#ifndef WORLEY_DIST_FNC
#define WORLEY_DIST_FNC distEuclidean
#endif

float2 worley2(float2 p){
    float2 n = floor( p );
    float2 f = frac( p );

    float distF1 = 1.0;
    float distF2 = 1.0;
    float2 off1, pos1;
    float2 off2, pos2;
    for( int j= -1; j <= 1; j++ ) {
        for( int i=-1; i <= 1; i++ ) {	
            float2  g = float2(i,j);
            float2  o = random2( n + g ) * WORLEY_JITTER;
            float2  p = g + o;
            float d = WORLEY_DIST_FNC(p, f);
            if (d < distF1) {
                distF2 = distF1;
                distF1 = d;
                off2 = off1;
                off1 = g;
                pos2 = pos1;
                pos1 = p;
            }
            else if (d < distF2) {
                distF2 = d;
                off2 = g;
                pos2 = p;
            }
        }
    }

    return float2(distF1, distF2);
}

float worley(float2 p){
    return 1.0-worley2(p).x;
}

float2 worley2(float3 p) {
    float3 n = floor( p );
    float3 f = frac( p );

    float distF1 = 1.0;
    float distF2 = 1.0;
    float3 off1, pos1;
    float3 off2, pos2;
    for( int k = -1; k <= 1; k++ ) {
        for( int j= -1; j <= 1; j++ ) {
            for( int i=-1; i <= 1; i++ ) {	
                float3  g = float3(i,j,k);
                float3  o = random3( n + g ) * WORLEY_JITTER;
                float3  p = g + o;
                float d = WORLEY_DIST_FNC(p, f);
                if (d < distF1) {
                    distF2 = distF1;
                    distF1 = d;
                    off2 = off1;
                    off1 = g;
                    pos2 = pos1;
                    pos1 = p;
                }
                else if (d < distF2) {
                    distF2 = d;
                    off2 = g;
                    pos2 = p;
                }
            }
        }
    }

    return float2(distF1, distF2);
}

float worley(float3 p){
    return 1.0-worley2(p).x;
}

#endif