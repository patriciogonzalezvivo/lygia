#include "map.cuh"

/*
contributors:  Inigo Quiles
description: Calculare Ambient Occlusion
use: <float> raymarchAO( in <float3> pos, in <float3> nor ) 
*/

#ifndef RAYMARCH_SAMPLES_AO
#define RAYMARCH_SAMPLES_AO 5
#endif

#ifndef RAYMARCH_MAP_FNC
#define RAYMARCH_MAP_FNC(POS) raymarchMap(POS)
#endif

#ifndef RAYMARCH_MAP_DISTANCE
#define RAYMARCH_MAP_DISTANCE w
#endif

#ifndef FNC_RAYMARCH_AO
#define FNC_RAYMARCH_AO

inline __host__ __device__ float raymarchAO(const float3& pos, const float3& nor ) {
    float occ = 0.0f;
    float sca = 1.0f;
    for( int i = 0; i < RAYMARCH_SAMPLES_AO; i++ ) {
        float hr = 0.01f + 0.12f * float(i) * 0.2f;
        float dd = RAYMARCH_MAP_FNC( hr * nor + pos ).RAYMARCH_MAP_DISTANCE;
        occ += (hr - dd) * sca;
        sca *= 0.9f;
        if( occ > 0.35f ) 
            break;
    }
    return clamp( 1.0f - 3.0f * occ, 0.0f, 1.0f) * (0.5f + 0.5f * nor.y);
}

#endif