#include "map.glsl"

/*
contributors:  Inigo Quiles
description: Cast a ray
use: <float> castRay( in <vec3> pos, in <vec3> nor ) 
*/

#ifndef RAYMARCH_SAMPLES
#define RAYMARCH_SAMPLES 64
#endif

#ifndef RAYMARCH_MIN_DIST
#define RAYMARCH_MIN_DIST 1.0
#endif

#ifndef RAYMARCH_MAX_DIST
#define RAYMARCH_MAX_DIST 20.0
#endif

#ifndef RAYMARCH_MIN_HIT_DIST
#define RAYMARCH_MIN_HIT_DIST 0.00001 * t
#endif

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef RAYMARCH_MAP_TYPE
#define RAYMARCH_MAP_TYPE vec4
#endif

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE a
#endif

#ifndef RAYMARCH_MAP_MATERIAL
#define RAYMARCH_MAP_MATERIAL rgb
#endif

#ifndef RAYMARCH_MAP_MATERIAL_TYPE
#define RAYMARCH_MAP_MATERIAL_TYPE vec3
#endif

#ifndef FNC_RAYMARCHCAST
#define FNC_RAYMARCHCAST

RAYMARCH_MAP_TYPE raymarchCast( in vec3 ro, in vec3 rd ) {
    float tmin = RAYMARCH_MIN_DIST;
    float tmax = RAYMARCH_MAX_DIST;
   
// #if defined(RAYMARCH_FLOOR)
//     float tp1 = (0.0-ro.y)/rd.y; if( tp1>0.0 ) tmax = min( tmax, tp1 );
//     float tp2 = (1.6-ro.y)/rd.y; if( tp2>0.0 ) { if( ro.y>1.6 ) tmin = max( tmin, tp2 );
//                                                  else           tmax = min( tmax, tp2 ); }
// #endif
    
    float t = tmin;
    RAYMARCH_MAP_MATERIAL_TYPE m = RAYMARCH_MAP_MATERIAL_TYPE(-1.0);
    for ( int i = 0; i < RAYMARCH_SAMPLES; i++ ) {
        RAYMARCH_MAP_TYPE res = RAYMARCH_MAP_FNC( ro + rd * t );
        if ( res.RAYMARCH_MAP_DISTANCE < RAYMARCH_MIN_HIT_DIST || t > tmax ) 
            break;
        t += res.RAYMARCH_MAP_DISTANCE;
        m = res.RAYMARCH_MAP_MATERIAL;
    }

    #if defined(RAYMARCH_BACKGROUND) || defined(RAYMARCH_FLOOR)
    if ( t > tmax ) 
        m = RAYMARCH_MAP_MATERIAL_TYPE(-1.0);
    #endif

    return RAYMARCH_MAP_TYPE( m, t );
}

#endif