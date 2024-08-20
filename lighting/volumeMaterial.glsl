/*
contributors: Shadi El Hajj
description: Volume Material Structure
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef STR_VOLUME_MATERIAL
#define STR_VOLUME_MATERIAL

struct VolumeMaterial {
    vec3    scattering;
    vec3    absorption;
    float   sdf;
};

#endif
