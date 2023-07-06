#include "../math/saturate.glsl"

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