#include "../math/saturate.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    This function calculates the reflection vector of a given vector and normal.
    It also takes into account the roughness of the material.
    If MATERIAL_ANISOTROPY is defined, it will also take into account the anisotropy direction.
    If MODEL_VERTEX_TANGENT is defined, it will use the tangentToWorld matrix to calculate the anisotropy direction.
use: <vec3> reflection(<vec3> vector, <vec3> normal, <float> roughness);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn reflection(_V: vec3f, _N: vec3f, _roughness: f32) -> vec3f {
        // Reflect
    let anisotropicT = MATERIAL_ANISOTROPY_DIRECTION;
    let anisotropicB = MATERIAL_ANISOTROPY_DIRECTION;

    anisotropicT = normalize(v_tangentToWorld * MATERIAL_ANISOTROPY_DIRECTION);
    anisotropicB = normalize(cross(v_tangentToWorld[2], anisotropicT));

    let anisotropyDirection = MATERIAL_ANISOTROPY >= 0.0 ? anisotropicB : anisotropicT;
    let anisotropicTangent = cross(anisotropyDirection, _V);
    let anisotropicNormal = cross(anisotropicTangent, anisotropyDirection);
    let bendFactor = abs(MATERIAL_ANISOTROPY) * saturate(5.0 * _roughness);
    let bentNormal = normalize(mix(_N, anisotropicNormal, bendFactor));
    return reflect(-_V, bentNormal);

    return reflect(-_V, _N);

}
