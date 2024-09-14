/*
contributors: Shadi El Hajj
description: Compute F0 for perceptual reflectance parameter. From Sébastien Lagarde's paper "Moving Frostbite to PBR".
use: <float> reflectance2f0(<float> reflectance)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef FNC_REFLECTANCE2F0
#define FNC_REFLECTANCE2F0
float reflectance2f0(const float reflectance) { return 0.16 * reflectance * reflectance; }
#endif