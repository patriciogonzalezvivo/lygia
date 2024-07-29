#include "map.glsl"
#include "../../math/saturate.glsl"

/*
contributors:  Inigo Quiles
description: Calculate Ambient Occlusion. See calcAO in https://www.shadertoy.com/view/lsKcDD
use: <float> raymarchAO( in <vec3> pos, in <vec3> nor ) 
examples:
    - /shaders/lighting_raymarching.frag
*/

#ifndef RAYMARCH_SAMPLES_AO
#define RAYMARCH_SAMPLES_AO 5
#endif

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE a
#endif

#ifndef FNC_RAYMARCHAO
#define FNC_RAYMARCHAO

float raymarchAO(in vec3 pos, in vec3 nor)
{
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < RAYMARCH_SAMPLES_AO; i++)
    {
        float h = 0.001 + 0.15 * float(i) / 4.0;
        float d = RAYMARCH_MAP_FNC(pos + h * nor).RAYMARCH_MAP_DISTANCE;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 1.5 * occ, 0.0, 1.0);
}

#endif