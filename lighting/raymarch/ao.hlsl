#include "map.hlsl"

/*
original_author:  Inigo Quiles
description: calculare Ambient Occlusion
use: <float> raymarchAO( in <float3> pos, in <float3> nor ) 
*/

#ifndef RAYMARCH_SAMPLES_AO
#define RAYMARCH_SAMPLES_AO 5
#endif

#ifndef FNC_RAYMARCHAO
#define FNC_RAYMARCHAO

float raymarchAO( in float3 pos, in float3 nor ) {
    float occ = 0.0;
    float sca = 1.0;
    for ( int i = 0; i < RAYMARCH_SAMPLES_AO; i++ ) {
        float hr = 0.01 + 0.12 * float(i) * 0.25;
        float3 aopos =  nor * hr + pos;
        float dd = raymarchMap( aopos ).a;
        occ += -(dd-hr)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );    
}

#endif