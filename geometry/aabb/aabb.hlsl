#include <cuda_runtime.h>

/*
contributors: Patricio Gonzalez Vivo
description: This file contains the definition of the AABB struct
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef STR_AABB
#define STR_AABB
struct AABB {
    float3 min;
    float3 max;
};
#endif