#include "../color/tonemap.wgsl"

#include "material.wgsl"
#include "light/new.wgsl"
#include "envMap.wgsl"
#include "specular.wgsl"
#include "fresnelReflection.wgsl"
#include "transparent.wgsl"

#include "ior/2eta.wgsl"
#include "ior/2f0.wgsl"

#include "reflection.wgsl"
#include "common/specularAO.wgsl"
#include "common/envBRDFApprox.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple glass shading model
use:
    - <vec4> glass(<Material> material)
options:
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughness (default on mobile)
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

// #define CAMERA_POSITION vec3(0.0, 0.0, -10.0)

const IBL_LUMINANCE: f32 = 1.0;

fn pbrGlass(mat: Material, shadingData: ShadingData) -> vec4f {
    // Shading Data
    // ------------
    vec3 No     = normalize(mat.normal - mat.normal_back); // Normal out is the difference between the front and back normals
    vec3 No     = mat.normal;                            // Normal out
    let eta = ior2eta(mat.ior);
    let f0 = vec3f(0.04, 0.04, 0.04);
    shadingData.N = mat.normal;
    shadingData.R = reflection(shadingData.V,  shadingData.N, mat.roughness);
    shadingData.roughness = mat.roughness; 
    shadingData.linearRoughness = mat.roughness;
    shadingData.specularColor = mat.albedo.rgb;
    shadingData.NoV = dot(No, shadingData.V);

    // Indirect Lights ( Image Based Lighting )
    // ----------------------------------------
    let E = envBRDFApprox(shadingData);
    let Gi = envMap(mat, shadingData) * E;

    let Fr = vec3f(0.0, 0.0, 0.0);
    Gi  += fresnelIridescentReflection(mat.normal, -shadingData.V, f0, vec3f(IOR_AIR),
        mat.ior, mat.thickness, mat.roughness, Fr);
    let Fr = fresnel(f0, shadingData.NoV);
    Gi  += fresnelReflection(shadingData.R, Fr) * (1.0-mat.roughness);

    let color = vec4f(0.0, 0.0, 0.0, 1.0);

    // Refraction
    color.rgb   += transparent(No, -shadingData.V, Fr, eta, mat.roughness);
    color.rgb   += Gi * IBL_LUMINANCE * mat.ambientOcclusion;

    // TODO: RaG
    //  - Add support for multiple lights
    // 
    {
        LightDirectional L = LightDirectionalNew();
        LightPoint L = LightPointNew();

        shadingData.L = L.direction;
        shadingData.H = normalize(L.direction + shadingData.V);
        shadingData.NoL = saturate(dot(shadingData.N, L.direction));
        shadingData.NoH = saturate(dot(shadingData.N, shadingData.H));
        let spec = specular(shadingData);

        color.rgb += L.color * spec;
    }

    return color;
}

fn pbrGlassa(mat: Material) -> vec4f {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - mat.position);
    return pbrGlass(mat, shadingData);
}
