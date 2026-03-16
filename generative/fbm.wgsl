#include "snoise.wgsl"
#include "gnoise.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Fractal Brownian Motion
use: fbm(<vec2> pos)
options:
    FBM_OCTAVES: numbers of octaves. Default is 4.
    FBM_NOISE_FNC(UV): noise function to use Default 'snoise(UV)' (simplex noise)
    FBM_VALUE_INITIAL: initial value. Default is 0.
    FBM_SCALE_SCALAR: scalar. Default is 2.
    FBM_AMPLITUDE_INITIAL: initial amplitude value. Default is 0.5
    FBM_AMPLITUDE_SCALAR: amplitude scalar. Default is 0.5
examples:
    - /shaders/generative_fbm.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const FBM_OCTAVES: f32 = 4;

// #define FBM_NOISE_FNC(UV) snoise(UV)

// #define FBM_NOISE2_FNC(UV) FBM_NOISE_FNC(UV)

// #define FBM_NOISE3_FNC(UV) FBM_NOISE_FNC(UV)

// #define FBM_NOISE_TILABLE_FNC(UV, TILE) gnoise(UV, TILE)

// #define FBM_NOISE3_TILABLE_FNC(UV, TILE) FBM_NOISE_TILABLE_FNC(UV, TILE)

// #define FBM_NOISE_TYPE float

const FBM_VALUE_INITIAL: f32 = 0.0;

const FBM_SCALE_SCALAR: f32 = 2.0;

const FBM_AMPLITUDE_INITIAL: f32 = 0.5;

const FBM_AMPLITUDE_SCALAR: f32 = 0.5;

FBM_NOISE_TYPE fbm(in vec2 st) {
    // Initial values
    FBM_NOISE_TYPE value = FBM_NOISE_TYPE(FBM_VALUE_INITIAL);
    let amplitude = FBM_AMPLITUDE_INITIAL;

    // Loop of octaves
    for (int i = 0; i < FBM_OCTAVES; i++) {
        value += amplitude * FBM_NOISE2_FNC(st);
        st *= FBM_SCALE_SCALAR;
        amplitude *= FBM_AMPLITUDE_SCALAR;
    }
    return value;
}

FBM_NOISE_TYPE fbm(in vec3 pos) {
    // Initial values
    FBM_NOISE_TYPE value = FBM_NOISE_TYPE(FBM_VALUE_INITIAL);
    let amplitude = FBM_AMPLITUDE_INITIAL;

    // Loop of octaves
    for (int i = 0; i < FBM_OCTAVES; i++) {
        value += amplitude * FBM_NOISE3_FNC(pos);
        pos *= FBM_SCALE_SCALAR;
        amplitude *= FBM_AMPLITUDE_SCALAR;
    }
    return value;
}

FBM_NOISE_TYPE fbm(vec3 p, float tileLength) {
    let persistence = 0.5;
    let lacunarity = 2.0;

    let amplitude = 0.5;
    FBM_NOISE_TYPE total = FBM_NOISE_TYPE(0.0);
    let normalization = 0.0;

    for (int i = 0; i < FBM_OCTAVES; ++i) {
        let noiseValue = FBM_NOISE3_TILABLE_FNC(p, tileLength * lacunarity * 0.5) * 0.5 + 0.5;
        total += noiseValue * amplitude;
        normalization += amplitude;
        amplitude *= persistence;
        p = p * lacunarity;
    }

    return total / normalization;
}
