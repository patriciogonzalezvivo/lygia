#include "shadingData/new.wgsl"
#include "specular/phong.wgsl"
#include "specular/blinnPhong.wgsl"
#include "specular/cookTorrance.wgsl"
#include "specular/gaussian.wgsl"
#include "specular/beckmann.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Calculate specular contribution
use:
    - specular(<vec3> L, <vec3> N, <vec3> V, <float> roughness[, <float> fresnel])
    - specular(<vec3> L, <vec3> N, <vec3> V, <float> NoV, <float> NoL, <float> roughness, <float> fresnel)
options:
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughness (default on mobile)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define SPECULAR_FNC specularBlinnPhongRoughness
// #define SPECULAR_FNC specularCookTorrance

fn specular(shadingData: ShadingData) -> vec3f { return vec3f(1.0, 1.0, 1.0) * SPECULAR_FNC(shadingData); }
