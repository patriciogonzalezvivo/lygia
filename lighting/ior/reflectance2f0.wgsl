/*
contributors: Shadi El Hajj
description: Compute F0 for perceptual reflectance parameter. From Sébastien Lagarde's paper "Moving Frostbite to PBR".
use: <float> reflectance2f0(<float> reflectance)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn reflectance2f0(reflectance: f32) -> f32 { return 0.16 * reflectance * reflectance; }
