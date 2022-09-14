#include "map.glsl"

/*
original_author:  Inigo Quiles
description: calculate soft shadows http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
use: <float> raymarchSoftshadow( in <vec3> ro, in <vec3> rd, in <float> mint, in <float> tmax) 
*/

#ifndef FNC_RAYMARCHSOFTSHADOW
#define FNC_RAYMARCHSOFTSHADOW

float raymarchSoftShadow( in vec3 ro, in vec3 rd, in float mint, in float tmax ) {
    float res = 1.0;
    float t = mint;
    for ( int i=0; i<16; i++ ) {
        float h = raymarchMap( ro + rd*t ).a;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( res<0.005 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

#endif