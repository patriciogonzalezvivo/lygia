/*
contributors: Patricio Gonzalez Vivo
description: Return the spherical harmonic value facing a normal direction
use: sphericalHarmonics( <vec3> normal)
options:
    SPHERICALHARMONICS_BANDS: 2 for RaspberryPi and WebGL for the rest is 3
    SCENE_SH_ARRAY: in GlslViewer is u_SH
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

const SPHERICALHARMONICS_BANDS: f32 = 1;
const SPHERICALHARMONICS_BANDS: f32 = 2;
const SPHERICALHARMONICS_BANDS: f32 = 3;

// #ifndef SCENE_SH_ARRAY
// #define SCENE_SH_ARRAY u_SH
// #endif

// #define SPHERICALHARMONICS_TONEMAP

fn sphericalHarmonics3(sh: array<vec3f, 9>, n: vec3f) -> vec3f {
    return SPHERICALHARMONICS_TONEMAP ( max(
           0.282095 * sh[0]
        + -0.488603 * sh[1] * (n.y)
        +  0.488603 * sh[2] * (n.z)
        + -0.488603 * sh[3] * (n.x)
        +  1.092548 * sh[4] * (n.y * n.x)
        + -1.092548 * sh[5] * (n.y * n.z)
        +  0.315392 * sh[6] * (3.0 * n.z * n.z - 1.0)
        + -1.092548 * sh[7] * (n.z * n.x)
        +  0.546274 * sh[8] * (n.x * n.x - n.y * n.y)
        , 0.0) );
}

fn sphericalHarmonics3a(n: vec3f) -> vec3f {
    return sphericalHarmonics(SCENE_SH_ARRAY, n);
    return vec3f(1.0);
}
