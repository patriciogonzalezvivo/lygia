#include "fresnel.glsl"
#include "envMap.glsl"
#include "../color/tonemap.glsl"
#include "sphericalHarmonics.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: resolve fresnel coeficient
use: 
    - <vec3> fresnel(const <vec3> f0, <float> LoH)
    - <vec3> fresnel(<vec3> _R, <vec3> _f0, <float> _NoV)
*/

#ifndef FNC_FRESNEL_REFLECTION
#define FNC_FRESNEL_REFLECTION

vec3 fresnelReflection(const in vec3 R, const in vec3 f0, const in float NoV) {
    vec3 frsnl = fresnel(f0, NoV);

    vec3 reflectColor = vec3(0.0);

    #if defined(FRESNEL_REFLECTION_FNC)
    reflection = FRESNEL_REFLECTION_FNC(R);

    // #elif defined(SCENE_SH_ARRAY)
    // reflectColor = tonemap( sphericalHarmonics(R) );

    #else
    reflectColor = envMap(R, 1.0, 0.001);

    #endif

    return reflectColor * frsnl;
}

vec3 fresnelIridescentReflection(vec3 R, vec3 f0, float NoV, float thickness, float ior0, float ior1, float ior2, float roughness) {
    vec3 frsnl = fresnelIridescent(f0, NoV, thickness, ior0, ior1, ior2, roughness);

    vec3 reflectColor = vec3(0.0);

    #if defined(FRESNEL_REFLECTION_FNC)
    reflection = FRESNEL_REFLECTION_FNC(R);

    // #elif defined(SCENE_SH_ARRAY)
    // reflectColor = tonemap( sphericalHarmonics(R) );

    #else
    reflectColor = envMap(R, roughness, 0.00001);

    #endif

    return reflectColor * frsnl;
}

vec3 fresnelIridescentReflection(vec3 normal, vec3 view, float thickness, float ior1, float ior2) {
    float cos0 = -dot(view, normal);
    const vec3 RGB = vec3(612.,549.,464.);
    // const vec3 RGB = vec3(650.0, 510.0, 475.0);
    
    // Schlick approximation
    float f0 = (ior2-1.)/(ior2+1.);
    float R = schlick(f0 * f0, 1.0, cos0);
    float T = 1.0-R;

    // Simpler formula based on:
    // https://www.alanzucconi.com/2017/07/25/the-mathematics-of-thin-film-interference/
    // Looks equivalent to my first attempt one above.
    float a = ior1/ior2;
    float cosi2 = sqrt(1. - a*a*(1. - cos0*cos0));
    vec3 shift = 4.*PI*(thickness/RGB)*ior2*cosi2 + HALF_PI;
    
    // Intensity of each color channel after interference
    vec3 irid = R*( 1. + T*( T + 2.*cos(shift) ) );
    
    // Reflection and background
    vec3 ref = envMap(reflect(view, normal), 0.0, 0.0);
    vec3 bak = envMap(refract(view, normal, 0.99), 0.0, 0.0);
    
    // Specular reflection
    vec3 spec = pow(ref, vec3(5.));
    
    // Final color
    return ref*irid + spec*irid + T*T*T*T*bak;
}


#ifdef STR_MATERIAL
vec3 fresnelReflection(const in Material _M) {
    #if defined(SHADING_MODEL_IRIDESCENCE)
    return fresnelIridescentReflection(_M.R, _M.f0, _M.NoV, _M.thickness, IOR_AIR, _M.ior.g, IOR_AIR, _M.roughness);
    // return fresnelIridescentReflection(_M.normal, _M.V, _M.thickness, IOR_AIR, _M.ior.g) * (1.0-_M.roughness);
    #else
    return fresnelReflection(_M.R, _M.f0, _M.NoV) * (1.0-_M.roughness);
    #endif
}
#endif

#endif