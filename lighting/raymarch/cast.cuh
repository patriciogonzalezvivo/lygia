#include "map.cuh"

/*
contributors:  Inigo Quiles
description: Cast a ray
use: <float> castRay( in <float3> pos, in <float3> nor ) 
*/

#ifndef RAYMARCH_SAMPLES
#define RAYMARCH_SAMPLES 64
#endif

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef RAYMARCH_MAP_TYPE
#define RAYMARCH_MAP_TYPE float4
#endif

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE w
#endif

#ifndef FNC_RAYMARCH_CAST
#define FNC_RAYMARCH_CAST

inline __host__ __device__ RAYMARCH_MAP_TYPE raymarchCast(const float3& ro, const float3& rd) {
    float tmin = 1.0;
    float tmax = 20.0;
    
// #if defined(RAYMARCH_FLOOR)
//     float tp1 = (0.0-ro.y)/rd.y; if( tp1>0.0 ) tmax = min( tmax, tp1 );
//     float tp2 = (1.6-ro.y)/rd.y; if( tp2>0.0 ) { if( ro.y>1.6 ) tmin = max( tmin, tp2 );
//                                                  else           tmax = min( tmax, tp2 ); }
// #endif

    float t = tmin;
    RAYMARCH_MAP_TYPE m = make_float4(-1.0f);
    for ( int i = 0; i < RAYMARCH_SAMPLES; i++ ) {
        float precis = 0.00001*t;
        RAYMARCH_MAP_TYPE res = RAYMARCH_MAP_FNC( ro + rd * t );
        if ( res.w < precis || t > tmax ) 
            break;
        t += res.w;
        m = res;
    }

    #if defined(RAYMARCH_BACKGROUND) || defined(RAYMARCH_FLOOR)
    if ( t > tmax ) 
        m = make_float4(-1.0);
    #endif

    m.RAYMARCH_MAP_DISTANCE = t;
    return m;
}

#endif