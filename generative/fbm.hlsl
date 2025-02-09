#include "snoise.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Fractal Brownian Motion
use: fbm(<float2|float3> pos)
options:
    FBM_OCTAVES: numbers of octaves. Default is 4.
    FBM_NOISE_FNC(POS_UV): noise function to use Default 'snoise(POS_UV)' (simplex noise)
    FBM_VALUE_INITIAL: initial value. Default is 0.
    FBM_SCALE_SCALAR: scalar. Default is 2.
    FBM_AMPLITUDE_INITIAL: initial amplitude value. Default is 0.5
    FBM_AMPLITUDE_SCALAR: amplitude scalar. Default is 0.5
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FBM_OCTAVES
#define FBM_OCTAVES 4
#endif

#ifndef FBM_NOISE_FNC
#define FBM_NOISE_FNC(POS_UV) snoise(POS_UV)
#endif

#ifndef FBM_NOISE2_FNC
#define FBM_NOISE2_FNC(POS_UV) FBM_NOISE_FNC(POS_UV)
#endif

#ifndef FBM_NOISE3_FNC
#define FBM_NOISE3_FNC(POS_UV) FBM_NOISE_FNC(POS_UV)
#endif

#ifndef FBM_NOISE_TYPE
#define FBM_NOISE_TYPE float
#endif


#ifndef FBM_VALUE_INITIAL
#define FBM_VALUE_INITIAL 0.0
#endif

#ifndef FBM_SCALE_SCALAR
#define FBM_SCALE_SCALAR 2.0
#endif

#ifndef FBM_AMPLITUDE_INITIAL
#define FBM_AMPLITUDE_INITIAL 0.5
#endif

#ifndef FBM_AMPLITUDE_SCALAR
#define FBM_AMPLITUDE_SCALAR 0.5
#endif

#ifndef FNC_FBM
#define FNC_FBM
FBM_NOISE_TYPE fbm(in float2 st) {
    // Initial values
    FBM_NOISE_TYPE value = FBM_VALUE_INITIAL;
    float amplitude = FBM_AMPLITUDE_INITIAL;

    // Loop of octaves
    for (int i = 0; i < FBM_OCTAVES; i++) {
        value += amplitude * FBM_NOISE2_FNC(st);
        st *= FBM_SCALE_SCALAR;
        amplitude *= FBM_AMPLITUDE_SCALAR;
    }
    return value;
}

FBM_NOISE_TYPE fbm(in float3 pos) {
    // Initial values
    FBM_NOISE_TYPE value = FBM_VALUE_INITIAL;
    float amplitude = FBM_AMPLITUDE_INITIAL;

    // Loop of octaves
    for (int i = 0; i < FBM_OCTAVES; i++) {
        value += amplitude * FBM_NOISE3_FNC(pos);
        pos *= FBM_SCALE_SCALAR;
        amplitude *= FBM_AMPLITUDE_SCALAR;
    }
    return value;
}
#endif
