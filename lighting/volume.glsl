/*
contributors: Shadi El Hajj
description: Volume Structure
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef STR_VOLUME
#define STR_VOLUME

struct Volume {
    vec3    scattering;
    vec3    absorption;
    float   sdf;
};

#endif
