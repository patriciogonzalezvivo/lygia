#include "random.glsl"
#include "../math/dist.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Worley noise. Returns vec2(F1, F2)
use: <vec2> worley2(<vec2|vec3> pos)
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

vec2 worley2(vec2 p){
    vec2 n = floor( p );
    vec2 f = fract( p );

    float distF1 = 1.0;
    float distF2 = 1.0;
    vec2 off1 = vec2(0.0); 
    vec2 pos1 = vec2(0.0);
    vec2 off2 = vec2(0.0);
    vec2 pos2 = vec2(0.0);
    for( int j= -1; j <= 1; j++ ) {
        for( int i=-1; i <= 1; i++ ) {	
            vec2  g = vec2(i,j);
            vec2  o = random2( n + g ) * WORLEY_JITTER;
            vec2  p = g + o;
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

    return vec2(distF1, distF2);
}

float worley(vec2 p){ return 1.0-worley2(p).x; }

vec2 worley2(vec3 p) {
    vec3 n = floor( p );
    vec3 f = fract( p );

    float distF1 = 1.0;
    float distF2 = 1.0;
    vec3 off1 = vec3(0.0);
    vec3 pos1 = vec3(0.0);
    vec3 off2 = vec3(0.0);
    vec3 pos2 = vec3(0.0);
    for( int k = -1; k <= 1; k++ ) {
        for( int j= -1; j <= 1; j++ ) {
            for( int i=-1; i <= 1; i++ ) {	
                vec3  g = vec3(i,j,k);
                vec3  o = random3( n + g ) * WORLEY_JITTER;
                vec3  p = g + o;
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

    return vec2(distF1, distF2);
}

float worley(vec3 p){ return 1.0-worley2(p).x; }

#endif