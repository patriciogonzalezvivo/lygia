#include "map.hlsl"

/*
contributors:  Inigo Quiles
description: Calculate soft shadows http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
use: <float> raymarchSoftshadow( in <float3> ro, in <float3> rd, in <float> tmin, in <float> tmax)
options:
    - RAYMARCHSOFTSHADOW_ITERATIONS: shadow quality
    - RAYMARCH_SHADOW_MIN_DIST: minimum shadow distance
    - RAYMARCH_SHADOW_MAX_DIST: maximum shadow distance
    - RAYMARCH_SHADOW_SOLID_ANGLE: light size
*/

#ifndef RAYMARCHSOFTSHADOW_ITERATIONS
#define RAYMARCHSOFTSHADOW_ITERATIONS 64
#endif

#ifndef RAYMARCH_SHADOW_MIN_DIST
#define RAYMARCH_SHADOW_MIN_DIST 0.01
#endif

#ifndef RAYMARCH_SHADOW_MAX_DIST
#define RAYMARCH_SHADOW_MAX_DIST 3.0
#endif

#ifndef RAYMARCH_SHADOW_SOLID_ANGLE
#define RAYMARCH_SHADOW_SOLID_ANGLE 0.1
#endif

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE a
#endif

#ifndef FNC_RAYMARCHSOFTSHADOW
#define FNC_RAYMARCHSOFTSHADOW

float raymarchSoftShadow( float3 ro, float3 rd, in float mint, in float maxt, float w ) {
    float res = 1.0;
    float t = mint;
    for (int i = 0; i < RAYMARCHSOFTSHADOW_ITERATIONS && t < maxt; i++)
    {
        float h = RAYMARCH_MAP_FNC(ro + t * rd).RAYMARCH_MAP_DISTANCE;
        res = min(res, h / (w * t));
        t += clamp(h, 0.005, 0.50);
        if (res < -1.0 || t > maxt)
            break;
    }
    res = max(res, -1.0);
    return 0.25 * (1.0 + res) * (1.0 + res) * (2.0 - res);
}

float raymarchSoftShadow( float3 ro, float3 rd, in float tmin, in float tmax) { return raymarchSoftShadow(ro, rd, RAYMARCH_SHADOW_MIN_DIST, RAYMARCH_SHADOW_MAX_DIST, RAYMARCH_SHADOW_SOLID_ANGLE); }
float raymarchSoftShadow( float3 ro, float3 rd) { return raymarchSoftShadow(ro, rd, RAYMARCH_SHADOW_MIN_DIST, RAYMARCH_SHADOW_MAX_DIST); }

#endif