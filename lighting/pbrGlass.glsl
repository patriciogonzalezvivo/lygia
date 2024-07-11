#include "../color/tonemap.glsl"

#include "material.glsl"
#include "light/new.glsl"
#include "envMap.glsl"
#include "specular.glsl"
#include "fresnelReflection.glsl"
#include "transparent.glsl"

#include "ior/2eta.glsl"
#include "ior/2f0.glsl"

#include "reflection.glsl"
#include "common/specularAO.glsl"
#include "common/envBRDFApprox.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple glass shading model
use:
    - <vec4> glass(<Material> material)
options:
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - SCENE_BACK_SURFACE: null
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_DIRECTION: null
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
examples:
    - /shaders/lighting_raymarching_glass.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#ifndef FNC_PBRGLASS
#define FNC_PBRGLASS

vec4 pbrGlass(const Material _mat) {
    
    // Cached
    Material M  = _mat;
    M.V         = normalize(CAMERA_POSITION - M.position);  // View
    M.R         = reflection(M.V, M.normal, M.roughness);   // Reflection
#if defined(SCENE_BACK_SURFACE)
    vec3 No     = normalize(M.normal - M.normal_back); // Normal out is the difference between the front and back normals
#else
    vec3 No     = M.normal;                            // Normal out
#endif
    M.NoV       = dot(No, M.V);                        // Normal . View

    vec3 eta    = ior2eta(M.ior);
    

    // Global Ilumination ( Image Based Lighting )
    // ------------------------
    vec3 E = envBRDFApprox(M.albedo.rgb, M);

    vec3 Gi = vec3(0.0, 0.0, 0.0);
    Gi  += envMap(M) * E;
    #if !defined(PLATFORM_RPI)
    // Gi  += fresnelReflection(M);

    #if defined(SHADING_MODEL_IRIDESCENCE)
    vec3 Fr = vec3(0.0);
    Gi  += fresnelIridescentReflection(M.normal, -M.V, M.f0, vec3(IOR_AIR), M.ior, M.thickness, M.roughness, Fr);
    #else
    vec3 Fr = fresnel(M.f0, M.NoV);
    Gi  += fresnelReflection(M.R, Fr) * (1.0-M.roughness);
    #endif

    #endif

    vec4 color  = vec4(0.0, 0.0, 0.0, 1.0);

    // Refraction
    color.rgb   += transparent(No, -M.V, Fr, eta, M.roughness);
    color.rgb   += Gi * IBL_LUMINANCE;

    // TODO: RaG
    //  - Add support for multiple lights
    // 
    {
        #if defined(LIGHT_DIRECTION)
        LightDirectional L = LightDirectionalNew();
        #elif defined(LIGHT_POSITION)
        LightPoint L = LightPointNew();
        #endif

        #if defined(LIGHT_DIRECTION) || defined(LIGHT_POSITION)
        // lightResolve(diffuseColor, specularColor, M, L, lightDiffuse, lightSpecular);
        vec3 spec = vec3( specular(L.direction, M.normal, M.V, M.roughness) );

        color.rgb += L.color * spec;

        #endif
    }

    return color;
}



#endif
