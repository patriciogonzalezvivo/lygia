/*
contributors: Shadi El Hajj
description: Classic Disney reparametrisation of roughness from Burley's paper "Physically Based Shading At Disney". Sébastien Lagarde's recommends clamping perceptual roughness to 0.045 in his paper "Moving Frostbite to PBR".
use: <float> perceptual2linearRoughness(<float> perceptualRoughness)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef MIN_PERCEPTUAL_ROUGHNESS
#define MIN_PERCEPTUAL_ROUGHNESS 0.045
#endif

#ifndef FNC_PERCEPTUAL_LINEAR_ROUGHNESS
#define FNC_PERCEPTUAL_LINEAR_ROUGHNESS

float perceptual2linearRoughness(float perceptualRoughness) {
    perceptualRoughness = clamp(perceptualRoughness, MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    return perceptualRoughness * perceptualRoughness;
}

#endif