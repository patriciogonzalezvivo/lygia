#include "random.glsl"
#include "../math/dist.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Worley noise
use: <vec2> worley(<vec2|vec3> pos)
options:
    - DIST_FNC: change the distance function, currently implemented are euclidean, manhattan, chebychev and minkowski
examples:
    - /shaders/generative_worley.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_WORLEY
#define FNC_WORLEY

#ifndef WORLEY_DIST_FNC
#define WORLEY_DIST_FNC distEuclidean
#endif

vec2 worley(vec2 p){
    vec2 n = floor( p );
    vec2 f = fract( p );

    float distF1 = 1.0;
    float distF2 = 1.0;
    vec2 off1, pos1;
    vec2 off2, pos2;
    for( int j= -1; j <= 1; j++ ) {
        for( int i=-1; i <= 1; i++ ) {	
            vec2  g = vec2(i,j);
            vec2  o = random2( n + g );
            vec2  point = g + o;
            float d = WORLEY_DIST_FNC(point, f);
            if (d < distF1) {
                distF2 = distF1;
                distF1 = d;
                off2 = off1;
                off1 = g;
                pos2 = pos1;
                pos1 = point;
            }
            else if (d < distF2) {
                distF2 = d;
                off2 = g;
                pos2 = point;
            }
        }
    }

    return vec2(distF1, distF2);
}

vec2 worley(vec3 p) {
    vec3 n = floor( p );
    vec3 f = fract( p );

    float distF1 = 1.0;
    float distF2 = 1.0;
    vec3 off1, pos1;
    vec3 off2, pos2;
    for( int k = -1; k <= 1; k++ ) {
        for( int j= -1; j <= 1; j++ ) {
            for( int i=-1; i <= 1; i++ ) {	
                vec3  g = vec3(i,j,k);
                vec3  o = random3( n + g );
                vec3  point = g + o;
                float d = WORLEY_DIST_FNC(point, f);
                if (d < distF1) {
                    distF2 = distF1;
                    distF1 = d;
                    off2 = off1;
                    off1 = g;
                    pos2 = pos1;
                    pos1 = point;
                }
                else if (d < distF2) {
                    distF2 = d;
                    off2 = g;
                    pos2 = point;
                }
            }
        }
    }

    return vec2(distF1, distF2);
}

#endif