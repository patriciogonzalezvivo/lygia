#include "../math/saturate.glsl"

#include "material.glsl"
#include "fresnelReflection.glsl"
#include "light/point.glsl"
#include "light/directional.glsl"

#include "reflection.glsl"
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
examples:
    - /shaders/lighting_raymarching_pbr.frag
*/

#ifndef CAMERA_POSITION
#define CAMERA_POSITION vec3(0.0, 0.0, -10.0)
#endif

#ifndef LIGHT_POSITION
#define LIGHT_POSITION  vec3(0.0, 10.0, -50.0)
#endif

#ifndef LIGHT_COLOR
#define LIGHT_COLOR     vec3(0.5, 0.5, 0.5)
#endif

#ifndef IBL_LUMINANCE
#define IBL_LUMINANCE   1.0
#endif

#ifndef FNC_PBR
#define FNC_PBR

vec4 pbr(const in Material _mat) {
    // Calculate Color
    vec3    diffuseColor = _mat.albedo.rgb * (vec3(1.0) - _mat.f0) * (1.0 - _mat.metallic);
    vec3    specularColor = mix(_mat.f0, _mat.albedo.rgb, _mat.metallic);

    vec3    N       = _mat.normal;                                  // Normal
    vec3    V       = normalize(CAMERA_POSITION - _mat.position);   // View
    float   NoV     = dot(N, V);                                    // Normal . View
    vec3    R       = reflection(V, N, _mat.roughness);

    // Ambient Occlusion
    // ------------------------
    float ssao = 1.0;
// #if defined(FNC_SSAO) && defined(SCENE_DEPTH) && defined(RESOLUTION) && defined(CAMERA_NEAR_CLIP) && defined(CAMERA_FAR_CLIP)
//     vec2 pixel = 1.0/RESOLUTION;
//     ssao = ssao(SCENE_DEPTH, gl_FragCoord.xy*pixel, pixel, 1.);
// #endif 

    // Global Ilumination ( Image Based Lighting )
    // ------------------------
    vec3 E = envBRDFApprox(specularColor, NoV, _mat.roughness);
    float diffuseAO = min(_mat.ambientOcclusion, ssao);

    vec3 Fr = vec3(0.0, 0.0, 0.0);
    Fr = envMap(R, _mat.roughness, _mat.metallic) * E * 2.0;
    #if !defined(PLATFORM_RPI)
    Fr += tonemap( fresnelReflection(R, _mat.f0, NoV) ) * _mat.metallic * (1.0-_mat.roughness) * 0.2;
    #endif
    Fr *= specularAO(NoV, diffuseAO, _mat.roughness);

    vec3 Fd = diffuseColor;
    #if defined(SCENE_SH_ARRAY)
    Fd *= tonemap( sphericalHarmonics(N) );
    #endif
    Fd *= diffuseAO;
    Fd *= (1.0 - E);

    // Local Ilumination
    // ------------------------
    vec3 lightDiffuse = vec3(0.0, 0.0, 0.0);
    vec3 lightSpecular = vec3(0.0, 0.0, 0.0);
    
    {
        #if defined(LIGHT_DIRECTION)
        float f0 = max(_mat.f0.r, max(_mat.f0.g, _mat.f0.b));
        lightDirectional(diffuseColor, specularColor, N, V, NoV, _mat.roughness, f0, _mat.shadow, lightDiffuse, lightSpecular);
        #elif defined(LIGHT_POSITION)
        float f0 = max(_mat.f0.r, max(_mat.f0.g, _mat.f0.b));
        lightPoint(diffuseColor, specularColor, N, V, NoV, _mat.roughness, f0, _mat.shadow, lightDiffuse, lightSpecular);
        #endif
    }
    
    // Final Sum
    // ------------------------
    vec4 color  = vec4(0.0, 0.0, 0.0, 1.0);
    // Diffuse
    color.rgb  += Fd * IBL_LUMINANCE;
    color.rgb  += lightDiffuse;     
    // Specular
    color.rgb  += Fr * IBL_LUMINANCE;
    color.rgb  += lightSpecular;    
    color.rgb  *= _mat.ambientOcclusion;
    color.rgb  += _mat.emissive;
    color.a     = _mat.albedo.a;

    return color;
}
#endif
