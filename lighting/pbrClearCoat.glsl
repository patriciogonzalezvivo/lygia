#include "../math/saturate.glsl"

#include "material.glsl"
#include "fresnelReflection.glsl"
#include "light/point.glsl"
#include "light/directional.glsl"

#include "ior/2f0.glsl"

#include "reflection.glsl"
#include "common/ggx.glsl"
#include "common/kelemen.glsl"
#include "common/specularAO.glsl"
#include "common/envBRDFApprox.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: simple PBR shading model
use: <vec4> pbr( <Material> _material ) 
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughnes (default on mobile)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
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

    vec3    N       = _mat.normal;                                  // Normal
    vec3    V       = normalize(CAMERA_POSITION - _mat.position);   // View
    float   NoV     = saturate(dot(N, V));                          // Normal . View
    vec3    f0      = ior2f0(_mat.ior);
    vec3    R       = reflection(V, N, _mat.roughness);

    #if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    // We want to use the geometric normal for the clear coat layer
    float clearCoatNoV      = clampNoV(dot(_mat.clearCoatNormal, V));
    vec3 clearCoatNormal    = _mat.clearCoatNormal;
    #else
    float clearCoatNoV      = NoV;
    vec3 clearCoatNormal    = N;
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
    vec3 E = envBRDFApprox(specularColor, NoV, _mat.roughness);

    // // This is a bit of a hack to pop the metalics
    // float specIntensity =   (2.0 * _mat.metallic) * 
    //                         saturate(-1.1 + NoV + _mat.metallic) *          // Fresnel
    //                         (_mat.metallic + (.95 - _mat.roughness) * 2.0); // make smaller highlights brighter

    float diffAO = min(_mat.ambientOcclusion, ssao);
    float specAO = specularAO(NoV, diffAO, _mat.roughness);

    vec3 Fr = vec3(0.0, 0.0, 0.0);
    Fr = envMap(R, _mat.roughness, _mat.metallic) * E * 2.0;
    #if !defined(PLATFORM_RPI)
    Fr += tonemap( fresnelReflection(R, f0, NoV) ) * _mat.metallic * (1.0-_mat.roughness) * 0.2;
    #endif
    Fr *= specAO;

    vec3 Fd = diffuseColor;
    #if defined(SCENE_SH_ARRAY)
    Fd *= tonemap( sphericalHarmonics(N) );
    #endif
    Fd *= diffAO;
    Fd *= (1.0 - E);

    vec3 Fc = fresnel(f0, clearCoatNoV) * _mat.clearCoat;
    vec3 attenuation = 1.0 - Fc;
    Fd *= attenuation;
    Fr *= attenuation;

    // vec3 clearCoatLobe = isEvaluateSpecularIBL(p, clearCoatNormal, V, clearCoatNoV);
    vec3 clearCoatR = reflection(V, clearCoatNormal, _mat.clearCoatRoughness);
    vec3 clearCoatE = envBRDFApprox(f0, clearCoatNoV, _mat.clearCoatRoughness);
    vec3 clearCoatLobe = vec3(0.0, 0.0, 0.0);
    clearCoatLobe += envMap(clearCoatR, _mat.clearCoatRoughness, 1.0) * clearCoatE * 3.;
    clearCoatLobe += tonemap( fresnelReflection(clearCoatR, f0, clearCoatNoV) ) * (1.0-_mat.clearCoatRoughness) * 0.2;
    Fr += clearCoatLobe * (specAO * _mat.clearCoat);

    vec4 color  = vec4(0.0, 0.0, 0.0, 1.0);
    color.rgb  += Fd * IBL_LUMINANCE;    // Diffuse
    color.rgb  += Fr * IBL_LUMINANCE;    // Specular

    // LOCAL ILUMINATION
    // ------------------------
    vec3 lightDiffuse = vec3(0.0, 0.0, 0.0);
    vec3 lightSpecular = vec3(0.0, 0.0, 0.0);
    
    {
        #if defined(LIGHT_DIRECTION)
        float f0 = max(_mat.f0.r, max(_mat.f0.g, _mat.f0.b));
        lightDirectional(diffuseColor, specularColor, N, V, NoV, _mat.roughness, f0, _mat.shadow, lightDiffuse, lightSpecular);
        #elif defined(LIGHT_POSITION)
        float f0 = max(_mat.f0.r, max(_mat.f0.g, _mat.f0.b));
        lightPoint(diffuseColor, specularColor, N, V, NoV, _mat.roughness , f0, _mat.shadow, lightDiffuse, lightSpecular);
        #endif
    }
    
    color.rgb  += lightDiffuse;     // Diffuse
    color.rgb  += lightSpecular;    // Specular

    // Clear Coat Local ilumination
    #if defined(LIGHT_DIRECTION) || defined(LIGHT_POSITION)

    #if defined(LIGHT_DIRECTION)
    vec3 L = normalize(LIGHT_DIRECTION);
    #elif defined(LIGHT_POSITION)
    vec3 L = normalize(LIGHT_POSITION - _mat.position);
    #endif

    vec3 H = normalize(V + L);
    float NoL = saturate(dot(N, L));
    float LoH = saturate(dot(L, H));

    #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    // If the material has a normal map, we want to use the geometric normal
    // instead to avoid applying the normal map details to the clear coat layer
    N = clearCoatNormal;
    float clearCoatNoH = saturate(dot(clearCoatNormal, H));
    #else
    float clearCoatNoH = saturate(dot(N, V));
    #endif

    // clear coat specular lobe
    float D         =   GGX(N, H, clearCoatNoH, _mat.clearCoatRoughness);
    vec3  F         =   fresnel(f0, LoH) * _mat.clearCoat;
    vec3  Fcc       =   F;
    vec3  clearCoat =   D * 
                        kelemen(LoH) * 
                        F;
    vec3  atten     =   (1.0 - Fcc);

    #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    // If the material has a normal map, we want to use the geometric normal
    // instead to avoid applying the normal map details to the clear coat layer
    float clearCoatNoL = saturate(dot(clearCoatNormal, L));
    color.rgb = color.rgb * atten * NoL + (clearCoat * clearCoatNoL * LIGHT_COLOR) * (LIGHT_INTENSITY * _mat.shadow);
    #else
    // color.rgb = color.rgb * atten + (clearCoat * LIGHT_COLOR) * (LIGHT_INTENSITY * NoL * _mat.shadow);
    color.rgb = color.rgb + (clearCoat * LIGHT_COLOR) * (LIGHT_INTENSITY * NoL * _mat.shadow);
    #endif

    #endif

    // Final
    color.rgb  *= _mat.ambientOcclusion;
    color.rgb  += _mat.emissive;
    color.a     = _mat.albedo.a;

    return color;
}
#endif