#include "map.glsl"

/*
original_author:  Inigo Quiles
description: cast a ray
use: <float> castRay( in <vec3> pos, in <vec3> nor ) 
*/

#ifndef RAYMARCH_SAMPLES
#define RAYMARCH_SAMPLES 64
#endif

#ifndef FNC_RAYMARCHCAST
#define FNC_RAYMARCHCAST

vec4 raymarchCast( in vec3 ro, in vec3 rd ) {
    float tmin = 1.0;
    float tmax = 20.0;
   
// #if defined(RAYMARCH_FLOOR)
//     float tp1 = (0.0-ro.y)/rd.y; if( tp1>0.0 ) tmax = min( tmax, tp1 );
//     float tp2 = (1.6-ro.y)/rd.y; if( tp2>0.0 ) { if( ro.y>1.6 ) tmin = max( tmin, tp2 );
//                                                  else           tmax = min( tmax, tp2 ); }
// #endif
    
    float t = tmin;
    vec3 m = vec3(-1.0);
    for ( int i = 0; i < RAYMARCH_SAMPLES; i++ ) {
        float precis = 0.00001*t;
        vec4 res = raymarchMap( ro + rd * t );
        if ( res.a < precis || t > tmax ) 
            break;
        t += res.a;
        m = res.rgb;
    }

    #if defined(RAYMARCH_BACKGROUND) || defined(RAYMARCH_FLOOR)
    if ( t>tmax ) 
        m = vec3(-1.0);
    #endif

    return vec4( m, t );
}

#endif