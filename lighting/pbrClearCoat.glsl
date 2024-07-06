#include "../math/saturate.glsl"
#include "../color/tonemap.glsl"

#include "material.glsl"
#include "fresnelReflection.glsl"
#include "envMap.glsl"
#include "light/new.glsl"
#include "light/resolve.glsl"
#include "sphericalHarmonics.glsl"

#include "ior/2f0.glsl"

#include "reflection.glsl"
#include "common/ggx.glsl"
#include "common/kelemen.glsl"
#include "common/specularAO.glsl"
#include "common/envBRDFApprox.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple PBR shading model
use: <vec4> pbr( <Material> _material )
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef CAMERA_POSITION
#define CAMERA_POSITION vec3(0.0, 0.0, -10.0);
#endif

#ifndef LIGHT_POSITION
#define LIGHT_POSITION  vec3(0.0, 10.0, -50.0)
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR     vec3(0.5, 0.5, 0.5)
#endif

#ifndef LIGHT_INTENSITY
#define LIGHT_INTENSITY 1.0
#endif

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#ifndef FNC_PBRCLEARCOAT
#define FNC_PBRCLEARCOAT

vec4 pbrClearCoat(const Material _mat) {
    // Calculate Color
    vec3    diffuseColor = _mat.albedo.rgb * (vec3(1.0) - _mat.f0) * (1.0 - _mat.metallic);
    vec3    specularColor = mix(_mat.f0, _mat.albedo.rgb, _mat.metallic);

    // Cached
    Material M  = _mat;
    M.V         = normalize(CAMERA_POSITION - M.position);  // View
    M.NoV       = dot(M.normal, M.V);                       // Normal . View
    M.R         = reflection(M.V, M.normal, M.roughness);   // Reflection

    vec3    f0      = ior2f0(M.ior);
    vec3    R       = reflection(M.V, M.normal, M.roughness);

    #if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    // We want to use the geometric normal for the clear coat layer
    float clearCoatNoV      = clampNoV(dot(M.clearCoatNormal, M.V));
    vec3 clearCoatNormal    = M.clearCoatNormal;
    #else
    float clearCoatNoV      = M.NoV;
    vec3 clearCoatNormal    = M.normal;
    #endif

    // Ambient Occlusion
    // ------------------------
    float ssao = 1.0;
// #if defined(FNC_SSAO) && defined(SCENE_DEPTH) && defined(RESOLUTION) && defined(CAMERA_NEAR_CLIP) && defined(CAMERA_FAR_CLIP)
//     vec2 pixel = 1.0/RESOLUTION;
//     ssao = ssao(SCENE_DEPTH, gl_FragCoord.xy*pixel, pixel, 1.);
// #endif 

    // Global Ilumination ( mage Based Lighting )
    // ------------------------
    vec3 E = envBRDFApprox(specularColor, M);

    // // This is a bit of a hack to pop the metalics
    // float specIntensity =   (2.0 * M.metallic) * 
    //                         saturate(-1.1 + NoV + M.metallic) *          // Fresnel
    //                         (M.metallic + (.95 - M.roughness) * 2.0); // make smaller highlights brighter

    float diffAO = min(M.ambientOcclusion, ssao);
    float specAO = specularAO(M, diffAO);

    vec3 Fr = vec3(0.0, 0.0, 0.0);
    Fr = envMap(M) * E;
    #if !defined(PLATFORM_RPI)
    Fr  += fresnelReflection(M);
    #endif
    Fr *= specAO;

    vec3 Fd = diffuseColor;
    #if defined(SCENE_SH_ARRAY)
    Fd *= tonemap( sphericalHarmonics(M.normal) );
    #endif
    Fd *= diffAO;
    Fd *= (1.0 - E);

    vec3 Fc = fresnel(f0, clearCoatNoV) * M.clearCoat;
    vec3 attenuation = 1.0 - Fc;
    Fd *= attenuation;
    Fr *= attenuation;

    // vec3 clearCoatLobe = isEvaluateSpecularIBL(p, clearCoatNormal, V, clearCoatNoV);
    vec3 clearCoatR = reflection(M.V, clearCoatNormal, M.clearCoatRoughness);
    vec3 clearCoatE = envBRDFApprox(f0, clearCoatNoV, M.clearCoatRoughness);
    vec3 clearCoatLobe = vec3(0.0, 0.0, 0.0);
    clearCoatLobe += envMap(clearCoatR, M.clearCoatRoughness, 1.0) * clearCoatE * 3.;
    clearCoatLobe += tonemap( fresnelReflection(clearCoatR, f0, clearCoatNoV) ) * (1.0-M.clearCoatRoughness) * 0.2;
    Fr += clearCoatLobe * (specAO * M.clearCoat);

    vec4 color  = vec4(0.0, 0.0, 0.0, 1.0);
    color.rgb  += Fd * IBL_LUMINANCE;    // Diffuse
    color.rgb  += Fr * IBL_LUMINANCE;    // Specular

    // LOCAL ILUMINATION
    // ------------------------
    vec3 lightDiffuse = vec3(0.0, 0.0, 0.0);
    vec3 lightSpecular = vec3(0.0, 0.0, 0.0);
    
    // TODO: 
    //  - Add support for multiple lights
    // 
    {
        #if defined(LIGHT_DIRECTION)
        LightDirectional L = LightDirectionalNew();
        #elif defined(LIGHT_POSITION)
        LightPoint L = LightPointNew();
        #endif

        #if defined(LIGHT_DIRECTION) || defined(LIGHT_POSITION)
        lightResolve(diffuseColor, specularColor, M, L, lightDiffuse, lightSpecular);

        color.rgb  += lightDiffuse;     // Diffuse
        color.rgb  += lightSpecular;    // Specular

        vec3  h     = normalize(M.V + L.direction);
        float NoH   = saturate(dot(M.normal, h));
        float NoL   = saturate(dot(M.normal, L.direction));
        float LoH   = saturate(dot(L.direction, h));

        #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
        // If the material has a normal map, we want to use the geometric normal
        // instead to avoid applying the normal map details to the clear coat layer
        N = clearCoatNormal;
        float clearCoatNoH = saturate(dot(clearCoatNormal, h));
        #else
        float clearCoatNoH = saturate(dot(M.normal, M.V));
        #endif

        // clear coat specular lobe
        float D         =   GGX(M.normal, h, clearCoatNoH, M.clearCoatRoughness);
        vec3  F         =   fresnel(f0, LoH) * M.clearCoat;

        vec3  Fcc       =   F;
        vec3  clearCoat =   vec3(D) * kelemen(LoH);// * F;
        vec3  atten     =   (1.0 - Fcc);

        #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
        // If the material has a normal map, we want to use the geometric normal
        // instead to avoid applying the normal map details to the clear coat layer
        float clearCoatNoL = saturate(dot(clearCoatNormal, L.direction));
        color.rgb = color.rgb * atten * NoL + (clearCoat * clearCoatNoL * L.color) * L.intensity;// * L.shadow;
        #else
        color.rgb = color.rgb * atten + (clearCoat * L.color) * (L.intensity * NoL);//(L.intensity * L.shadow * NoL);
        #endif

        #endif
    }
    
    // Final
    color.rgb  *= M.ambientOcclusion;
    color.rgb  += M.emissive;
    color.a     = M.albedo.a;

    return color;
}
#endif