#include "map.wgsl"

/*
contributors:  Inigo Quiles
description: Calculate soft shadows http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
use: <float> raymarchSoftshadow( in <vec3> ro, in <vec3> rd ) 
options:
    - RAYMARCHSOFTSHADOW_ITERATIONS: shadow quality
    - RAYMARCH_SHADOW_MIN_DIST: minimum shadow distance
    - RAYMARCH_SHADOW_MAX_DIST: maximum shadow distance
    - RAYMARCH_SHADOW_SOLID_ANGLE: light size
examples:
    - /shaders/lighting_raymarching.frag
*/

const RAYMARCH_MAX_DIST: f32 = 20.0;

const RAYMARCH_SOFTSHADOW_ITERATIONS: f32 = 64;

const RAYMARCH_SHADOW_MIN_DIST: f32 = 0.005;

// #define RAYMARCH_SHADOW_MAX_DIST RAYMARCH_MAX_DIST

const RAYMARCH_SHADOW_SOLID_ANGLE: f32 = 0.1;

fn raymarchSoftShadow(ro: vec3f, rd: vec3f) -> f32 {
    let mint = RAYMARCH_SHADOW_MIN_DIST;
    let maxt = RAYMARCH_SHADOW_MAX_DIST;
    let w = RAYMARCH_SHADOW_SOLID_ANGLE;

    let res = 1.0;
    let t = mint;
    for (int i = 0; i < RAYMARCH_SOFTSHADOW_ITERATIONS; i++) {
        if (t >= maxt)
            break;
        let h = RAYMARCH_MAP_FNC(ro + t * rd).sdf;
        res = min(res, h / (w * t));

        t += clamp(h, RAYMARCH_SHADOW_MIN_DIST, RAYMARCH_SHADOW_MAX_DIST);
        if (res < -1.0 || t > maxt)
            break;
    }
    res = max(res, -1.0);
    return 0.25 * (1.0 + res) * (1.0 + res) * (2.0 - res);
}
