#include "map.wgsl"
#include "../../math/saturate.wgsl"

/*
contributors:  Inigo Quiles
description: Calculate Ambient Occlusion. See calcAO in https://www.shadertoy.com/view/lsKcDD
use: <float> raymarchAO( in <vec3> pos, in <vec3> nor ) 
examples:
    - /shaders/lighting_raymarching.frag
*/

fn raymarchAO(pos: vec3f, nor: vec3f) -> f32 {
    const RAYMARCH_AO_SAMPLES: f32 = 5;
    const RAYMARCH_AO_INTENSITY: f32 = 1.0;
    const RAYMARCH_AO_MIN_DIST: f32 = 0.001;
    const RAYMARCH_AO_MAX_DIST: f32 = 0.2;
    const RAYMARCH_AO_FALLOFF: f32 = 0.95;
    let occ = 0.0;
    let sca = 1.0;
    let samplesFactor = 1.0 / float(RAYMARCH_AO_SAMPLES-1);
    for (int i = 0; i < RAYMARCH_AO_SAMPLES; i++) {
        let h = RAYMARCH_AO_MIN_DIST + RAYMARCH_AO_MAX_DIST * float(i) * samplesFactor;
        let d = RAYMARCH_MAP_FNC(pos + h * nor).sdf;
        occ += (h - d) * sca;
        sca *= RAYMARCH_AO_FALLOFF;
    }
    return saturate(1.0 - RAYMARCH_AO_INTENSITY * occ);
}
