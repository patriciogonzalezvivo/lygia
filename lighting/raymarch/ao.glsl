#include "map.glsl"
#include "../../math/saturate.glsl"

/*
contributors:  Inigo Quiles
description: Calculate Ambient Occlusion. See calcAO in https://www.shadertoy.com/view/lsKcDD
use: <float> raymarchAO( in <vec3> pos, in <vec3> nor ) 
examples:
    - /shaders/lighting_raymarching.frag
*/

#ifndef RAYMARCH_AO_SAMPLES
#define RAYMARCH_AO_SAMPLES 5
#endif

#ifndef RAYMARCH_AO_INTENSITY
#define RAYMARCH_AO_INTENSITY 1.0
#endif

#ifndef RAYMARCH_AO_MIN_DIST
#define RAYMARCH_AO_MIN_DIST 0.001
#endif

#ifndef RAYMARCH_AO_MAX_DIST
#define RAYMARCH_AO_MAX_DIST 0.2
#endif

#ifndef RAYMARCH_AO_FALLOFF
#define RAYMARCH_AO_FALLOFF 0.95
#endif

#ifndef FNC_RAYMARCH_AO
#define FNC_RAYMARCH_AO

float raymarchAO(in vec3 pos, in vec3 nor) {
    float occ = 0.0;
    float sca = 1.0;
    const float samplesFactor = 1.0 / float(RAYMARCH_AO_SAMPLES-1);
    for (int i = 0; i < RAYMARCH_AO_SAMPLES; i++) {
        float h = RAYMARCH_AO_MIN_DIST + RAYMARCH_AO_MAX_DIST * float(i) * samplesFactor;
        float d = RAYMARCH_MAP_FNC(pos + h * nor).sdf;
        occ += (h - d) * sca;
        sca *= RAYMARCH_AO_FALLOFF;
    }
    return saturate(1.0 - RAYMARCH_AO_INTENSITY * occ);
}

#endif