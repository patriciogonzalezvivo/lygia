#include "../math/saturate.wgsl"
#include "../color/tonemap.wgsl"

#include "common/ggx.wgsl"
#include "common/kelemen.wgsl"

#include "shadingData/new.wgsl"
#include "material.wgsl"
#include "envMap.wgsl"
#include "fresnelReflection.wgsl"
#include "sphericalHarmonics.wgsl"
#include "light/new.wgsl"
#include "light/resolve.wgsl"

#include "reflection.wgsl"
#include "common/specularAO.wgsl"
#include "common/envBRDFApprox.wgsl"

#include "ior/2f0.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Simple PBR shading model
use: <vec4> pbr( <Material> material )
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughness (default on mobile)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define CAMERA_POSITION vec3(0.0, 0.0, -10.0)

// #define LIGHT_POSITION  vec3(0.0, 10.0, -50.0)

// #define LIGHT_COLOR     vec3(0.5, 0.5, 0.5)

const LIGHT_INTENSITY: f32 = 1.0;

const IBL_LUMINANCE: f32 = 1.0;

fn pbrClearCoat(mat: Material, shadingData: ShadingData) -> vec4f {
    shadingDataNew(mat, shadingData);

    let f0 = ior2f0(mat.ior);
    let R = reflection(shadingData.V, mat.normal, mat.roughness);

    // We want to use the geometric normal for the clear coat layer
    let clearCoatNoV = clampNoV(dot(mat.clearCoatNormal, shadingData.V));
    let clearCoatNormal = mat.clearCoatNormal;
    let clearCoatNoV = shadingData.NoV;
    let clearCoatNormal = mat.normal;

    // Indirect Lights ( Image Based Lighting )
    // ----------------------------------------
    let E = envBRDFApprox(shadingData);

    let diffAO = mat.ambientOcclusion;
    let specAO = specularAO(mat, shadingData, diffAO);

    let Fr = vec3f(0.0, 0.0, 0.0);
    Fr = envMap(mat, shadingData) * E;
    Fr  += fresnelReflection(mat, shadingData);
    Fr *= specAO;

    let Fd = shadingData.diffuseColor;
    Fd *= tonemap( sphericalHarmonics(mat.normal) );
    Fd *= diffAO;
    Fd *= (1.0 - E);

    let Fc = fresnel(f0, clearCoatNoV) * mat.clearCoat;
    let attenuation = 1.0 - Fc;
    Fd *= attenuation;
    Fr *= attenuation;

    // vec3 clearCoatLobe = isEvaluateSpecularIBL(p, clearCoatNormal, V, clearCoatNoV);
    let clearCoatR = reflection(shadingData.V, clearCoatNormal, mat.clearCoatRoughness);
    let clearCoatE = envBRDFApprox(f0, clearCoatNoV, mat.clearCoatRoughness);
    let clearCoatLobe = vec3f(0.0, 0.0, 0.0);
    clearCoatLobe += envMap(clearCoatR, mat.clearCoatRoughness, 1.0) * clearCoatE * 3.;
    clearCoatLobe += tonemap( fresnelReflection(clearCoatR, f0, clearCoatNoV) ) * (1.0-mat.clearCoatRoughness) * 0.2;
    Fr += clearCoatLobe * (specAO * mat.clearCoat);

    let color = vec4f(0.0, 0.0, 0.0, 1.0);
    color.rgb  += Fd * IBL_LUMINANCE;    // Diffuse
    color.rgb  += Fr * IBL_LUMINANCE;    // Specular

    // Direct Lights
    // -------------
    // TODO: 
    //  - Add support for multiple lights
    // 
    {
        LightDirectional L = LightDirectionalNew();
        LightPoint L = LightPointNew();

        lightResolve(L, mat, shadingData);

        color.rgb  += shadingData.directDiffuse;     // Diffuse
        color.rgb  += shadingData.directSpecular;    // Specular

        let h = normalize(shadingData.V + L.direction);
        let NoH = saturate(dot(mat.normal, h));
        let NoL = saturate(dot(mat.normal, L.direction));
        let LoH = saturate(dot(L.direction, h));

        // If the material has a normal map, we want to use the geometric normal
        // instead to avoid applying the normal map details to the clear coat layer
        N = clearCoatNormal;
        let clearCoatNoH = saturate(dot(clearCoatNormal, h));
        let clearCoatNoH = saturate(dot(mat.normal, shadingData.V));

        // clear coat specular lobe
        let D = GGX(mat.normal, h, clearCoatNoH, mat.clearCoatRoughness);
        let F = fresnel(f0, LoH) * mat.clearCoat;

        let Fcc = F;
        let clearCoat = vec3f(D, D, D) * kelemen(LoH);// * F;
        let atten = (1.0 - Fcc);

        // If the material has a normal map, we want to use the geometric normal
        // instead to avoid applying the normal map details to the clear coat layer
        let clearCoatNoL = saturate(dot(clearCoatNormal, L.direction));
        color.rgb = color.rgb * atten * NoL + (clearCoat * clearCoatNoL * L.color) * L.intensity;// * L.shadow;
        color.rgb = color.rgb * atten + (clearCoat * L.color) * (L.intensity * NoL);//(L.intensity * L.shadow * NoL);

    }
    
    // Final
    color.rgb  *= mat.ambientOcclusion;
    color.rgb  += mat.emissive;
    color.a     = mat.albedo.a;

    return color;
}

fn pbrClearCoata(mat: Material) -> vec4f {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - mat.position);
    return pbrClearCoat(mat, shadingData);
}
