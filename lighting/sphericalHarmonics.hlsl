/*
contributors: Patricio Gonzalez Vivo
description: Return the spherical harmonic value facing a normal direction
use: sphericalHarmonics( <float3> normal)
options:
        SPHERICALHARMONICS_BANDS: 2 for RaspberryPi and WebGL for the rest is 3
        SCENE_SH_ARRAY: in GlslViewer is u_SH
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SPHERICALHARMONICS_BANDS
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
#define SPHERICALHARMONICS_BANDS           2
#else
#define SPHERICALHARMONICS_BANDS           3
#endif
#endif

// #ifndef SCENE_SH_ARRAY
// #define SCENE_SH_ARRAY u_SH
// #endif

#ifndef SPHERICALHARMONICS_TONEMAP 
#define SPHERICALHARMONICS_TONEMAP
#endif

#ifndef FNC_SPHERICALHARMONICS
#define FNC_SPHERICALHARMONICS

float3 sphericalHarmonics(const float3 n) {
#ifdef SCENE_SH_ARRAY
    return SPHERICALHARMONICS_TONEMAP ( max(
           0.282095 * SCENE_SH_ARRAY[0]
#if SPHERICALHARMONICS_BANDS >= 2
        + -0.488603 * SCENE_SH_ARRAY[1] * (n.y)
        +  0.488603 * SCENE_SH_ARRAY[2] * (n.z)
        + -0.488603 * SCENE_SH_ARRAY[3] * (n.x)
#endif
#if SPHERICALHARMONICS_BANDS >= 3
        +  1.092548 * SCENE_SH_ARRAY[4] * (n.y * n.x)
        + -1.092548 * SCENE_SH_ARRAY[5] * (n.y * n.z)
        +  0.315392 * SCENE_SH_ARRAY[6] * (3.0 * n.z * n.z - 1.0)
        + -1.092548 * SCENE_SH_ARRAY[7] * (n.z * n.x)
        +  0.546274 * SCENE_SH_ARRAY[8] * (n.x * n.x - n.y * n.y)
#endif
        , 0.0) );
#else
    return float3(1.0, 1.0, 1.0);
#endif
}

#endif