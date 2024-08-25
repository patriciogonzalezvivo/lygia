/*
contributors: Patricio Gonzalez Vivo
description: Calculate diffuse contribution using Oren and Nayar equation https://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model
use:
    - <float> diffuseOrenNayar(<vec3> light, <vec3> normal, <vec3> view, <float> roughness)
    - <float> diffuseOrenNayar(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_DIFFUSE_ORENNAYAR
#define FNC_DIFFUSE_ORENNAYAR

float diffuseOrenNayar(ShadingData shadingData) {
    float LoV = dot(shadingData.L, shadingData.V);
    
    float s = LoV - shadingData.NoL * shadingData.NoV;
    float t = mix(1.0, max(shadingData.NoL, shadingData.NoV), step(0.0, s));

    float sigma2 = shadingData.linearRoughness * shadingData.linearRoughness;
    float A = 1.0 + sigma2 * (1.0 / (sigma2 + 0.13) + 0.5 / (sigma2 + 0.33));
    float B = 0.45 * sigma2 / (sigma2 + 0.09);

    return max(0.0, shadingData.NoL) * (A + B * s / t);
}

#endif