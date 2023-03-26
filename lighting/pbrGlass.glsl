#include "material.glsl"
#include "ior.glsl"
#include "specular.glsl"
#include "fresnelReflection.glsl"

#include "ior/2eta.glsl"
#include "ior/2f0.glsl"

#include "reflection.glsl"
#include "common/specularAO.glsl"
#include "common/envBRDFApprox.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: simple glass shading model
use: 
    - <vec4> glass(<Material> material) 
    
options:
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - SCENE_BACK_SURFACE: 
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_DIRECTION: 
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
examples:
    - /shaders/lighting_raymarching_glass.frag
*/

#ifndef LIGHT_COLOR
#define LIGHT_COLOR     vec3(1.0)
#endif

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#ifndef FNC_PBRGLASS
#define FNC_PBRGLASS

vec4 pbrGlass(const Material _mat) {
    vec3    V       = normalize(CAMERA_POSITION - _mat.position);   // View
    vec3    N       = _mat.normal;                                  // Normal front
    vec3    No      = _mat.normal;                                  // Normal out
#if defined(SCENE_BACK_SURFACE)
            No      = normalize( N - _mat.normal_back );
#endif

    float roughness = _mat.roughness;

    vec3    f0      = ior2f0(_mat.ior);
    vec3    eta     = ior2eta(_mat.ior);
    vec3    Re      = reflection(V, N, roughness);
    vec3    RaG     = refract(-V, No, eta.g);
    #if !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI)
    vec3    RaR     = refract(-V, No, eta.r);
    vec3    RaB     = refract(-V, No, eta.b);
    #endif

    float   NoV     = dot(N, V);                                    // Normal . View

    // Global Ilumination ( mage Based Lighting )
    // ------------------------
    vec3 E = envBRDFApprox(_mat.albedo.rgb, NoV, roughness);

    vec3 Fr = vec3(0.0, 0.0, 0.0);
    Fr = envMap(Re, roughness) * E;
    #if !defined(PLATFORM_RPI)
    Fr += tonemap( fresnelReflection(Re, _mat.f0, NoV) ) * (1.0-roughness) * 0.2;
    #endif

    vec4 color  = vec4(0.0, 0.0, 0.0, 1.0);
    color.rgb   = envMap(RaG, roughness);
    #if !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI)
    color.r     = envMap(RaR, roughness).r;
    color.b     = envMap(RaB, roughness).b;
    #endif
    // color.rgb   *= exp( -_mat.thickness * 200.0);
    color.rgb   += Fr * IBL_LUMINANCE;

    #if defined(LIGHT_DIRECTION)
    color.rgb += LIGHT_COLOR * specular(normalize(LIGHT_DIRECTION), N, V, roughness) * _mat.shadow;
    #elif defined(LIGHT_POSITION)
    color.rgb += LIGHT_COLOR * specular(normalize(LIGHT_POSITION - position), N, V, roughness) * _mat.shadow;
    #endif

    return color;
}

#endif