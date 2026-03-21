/*
contributors: Shadi El Hajj
description: Classic Disney reparametrisation of roughness from Burley's paper "Physically Based Shading At Disney". Sébastien Lagarde's recommends clamping perceptual roughness to 0.045 in his paper "Moving Frostbite to PBR".
use: <float> perceptual2linearRoughness(<float> perceptualRoughness)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

const MIN_PERCEPTUAL_ROUGHNESS: f32 = 0.045;

fn perceptual2linearRoughness(perceptualRoughness: f32) -> f32 {
    perceptualRoughness = clamp(perceptualRoughness, MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    return perceptualRoughness * perceptualRoughness;
}
