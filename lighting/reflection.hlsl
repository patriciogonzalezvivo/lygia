/*
contributors: Patricio Gonzalez Vivo
description: |
    This function calculates the reflection vector of a given vector and normal.
    It also takes into account the roughness of the material.
    If MATERIAL_ANISOTROPY is defined, it will also take into account the anisotropy direction.
    If MODEL_VERTEX_TANGENT is defined, it will use the tangentToWorld matrix to calculate the anisotropy direction.
use: <float3> reflection(<float3> vector, <float3> normal, <float> roughness);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_REFLECTION
#define FNC_REFLECTION

float3 reflection(float3 _V, float3 _N, float _roughness) {
        // Reflect
#ifdef MATERIAL_ANISOTROPY
    float3  anisotropicT = MATERIAL_ANISOTROPY_DIRECTION;
    float3  anisotropicB = MATERIAL_ANISOTROPY_DIRECTION;

#ifdef MODEL_VERTEX_TANGENT
    anisotropicT = normalize(v_tangentToWorld * MATERIAL_ANISOTROPY_DIRECTION);
    anisotropicB = normalize(cross(v_tangentToWorld[2], anisotropicT));
#endif

    float3  anisotropyDirection = MATERIAL_ANISOTROPY >= 0.0 ? anisotropicB : anisotropicT;
    float3  anisotropicTangent  = cross(anisotropyDirection, _V);
    float3  anisotropicNormal   = cross(anisotropicTangent, anisotropyDirection);
    float bendFactor            = abs(MATERIAL_ANISOTROPY) * saturate(5.0 * _roughness);
    float3  bentNormal          = normalize(lerp(_N, anisotropicNormal, bendFactor));
    return reflect(-_V, bentNormal);
#else
    return reflect(-_V, _N);
#endif

}

#endif