#include "../math/saturate.glsl"

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

#ifndef FNC_REFLECTION
#define FNC_REFLECTION

vec3 reflection(const in vec3 _V, const in vec3 _N, const in float _roughness) {
        // Reflect
#ifdef MATERIAL_ANISOTROPY
    vec3  anisotropicT = MATERIAL_ANISOTROPY_DIRECTION;
    vec3  anisotropicB = MATERIAL_ANISOTROPY_DIRECTION;

    #ifdef MODEL_VERTEX_TANGENT
    anisotropicT = normalize(v_tangentToWorld * MATERIAL_ANISOTROPY_DIRECTION);
    anisotropicB = normalize(cross(v_tangentToWorld[2], anisotropicT));
    #endif

    vec3  anisotropyDirection = MATERIAL_ANISOTROPY >= 0.0 ? anisotropicB : anisotropicT;
    vec3  anisotropicTangent  = cross(anisotropyDirection, _V);
    vec3  anisotropicNormal   = cross(anisotropicTangent, anisotropyDirection);
    float bendFactor          = abs(MATERIAL_ANISOTROPY) * saturate(5.0 * _roughness);
    vec3  bentNormal          = normalize(mix(_N, anisotropicNormal, bendFactor));
    return reflect(-_V, bentNormal);
#else

    return reflect(-_V, _N);
#endif

}

#endif