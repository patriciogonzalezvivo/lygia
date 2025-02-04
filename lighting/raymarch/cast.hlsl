#include "map.hlsl"
#include "../material/new.hlsl"

/*
contributors:  Inigo Quiles
description: Cast a ray
use: <float> castRay( in <float3> pos, in <float3> nor ) 
*/

#ifndef RAYMARCH_SAMPLES
#define RAYMARCH_SAMPLES 256
#endif

#ifndef RAYMARCH_MIN_DIST
#define RAYMARCH_MIN_DIST 0.1
#endif

#ifndef RAYMARCH_MAX_DIST
#define RAYMARCH_MAX_DIST 20.0
#endif

#ifndef RAYMARCH_MIN_HIT_DIST
#define RAYMARCH_MIN_HIT_DIST 0.00001 * t
#endif

#ifndef FNC_RAYMARCH_CAST
#define FNC_RAYMARCH_CAST

Material raymarchCast( in float3 ro, in float3 rd ) {
    float tmin = RAYMARCH_MIN_DIST;
    float tmax = RAYMARCH_MAX_DIST;
   
    float t = tmin;
    Material m = materialNew();
    m.valid = false;
    for (int i = 0; i < RAYMARCH_SAMPLES; i++) {
        Material res = RAYMARCH_MAP_FNC(ro + rd * t);
#ifdef RAYMARCH_ABS_DIST
        float dist = abs(res.sdf);
#else
        float dist = res.sdf;
#endif
        if (dist < RAYMARCH_MIN_HIT_DIST || t > tmax) break;
        m = res;
        t += res.sdf;
    }

    #if defined(RAYMARCH_BACKGROUND) || defined(RAYMARCH_FLOOR)
    if ( t > tmax )
        m.valid = false;
    #endif

    m.sdf = t;
    return m;
}

#endif