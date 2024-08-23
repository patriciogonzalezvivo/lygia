#include "map.hlsl"

/*
contributors:  Inigo Quiles
description: Calculate soft shadows http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
use: <float> raymarchSoftshadow( in <float3> ro, in <float3> rd ) 
options:
    - RAYMARCHSOFTSHADOW_ITERATIONS: shadow quality
    - RAYMARCH_SHADOW_MIN_DIST: minimum shadow distance
    - RAYMARCH_SHADOW_MAX_DIST: maximum shadow distance
    - RAYMARCH_SHADOW_SOLID_ANGLE: light size
examples:
    - /shaders/lighting_raymarching.frag
*/

#ifndef RAYMARCHSOFTSHADOW_ITERATIONS
#define RAYMARCHSOFTSHADOW_ITERATIONS 64
#endif

#ifndef RAYMARCH_SHADOW_MIN_DIST
#define RAYMARCH_SHADOW_MIN_DIST 0.005
#endif

#ifndef RAYMARCH_SHADOW_MAX_DIST
#define RAYMARCH_SHADOW_MAX_DIST RAYMARCH_MAX_DIST
#endif

#ifndef RAYMARCH_SHADOW_SOLID_ANGLE
#define RAYMARCH_SHADOW_SOLID_ANGLE 0.1
#endif

#ifndef FNC_RAYMARCH_SOFTSHADOW
#define FNC_RAYMARCH_SOFTSHADOW

float raymarchSoftShadow(float3 ro, float3 rd) {
    const float mint = RAYMARCH_SHADOW_MIN_DIST;
    const float maxt = RAYMARCH_SHADOW_MAX_DIST;
    const float w = RAYMARCH_SHADOW_SOLID_ANGLE;

    float res = 1.0;
    float t = mint;
    for (int i = 0; i < RAYMARCHSOFTSHADOW_ITERATIONS; i++) {
        if (t >= maxt)
            break;
        float h = RAYMARCH_MAP_FNC(ro + t * rd).sdf;
        res = min(res, h / (w * t));

        t += clamp(h, RAYMARCH_SHADOW_MIN_DIST, RAYMARCH_SHADOW_MAX_DIST);
        if (res < -1.0 || t > maxt)
            break;
    }
    res = max(res, -1.0);
    return 0.25 * (1.0 + res) * (1.0 + res) * (2.0 - res);
}

#endif