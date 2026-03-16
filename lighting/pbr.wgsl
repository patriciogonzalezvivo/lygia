// #define DIFFUSE_FNC diffuseLambertConstant

// #define SPECULAR_FNC specularCookTorrance

#include "../math/saturate.wgsl"
#include "shadingData/new.wgsl"
#include "material.wgsl"
#include "light/new.wgsl"
#include "light/resolve.wgsl"
#include "light/iblEvaluate.wgsl"

/*
contributors: [Patricio Gonzalez Vivo, Shadi El Hajj]
description: Simple PBR shading model
use: <vec4> pbr( <Material> material )
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SPECULAR_FNC: specularGaussian, specularBeckmann, specularCookTorrance (default), specularPhongRoughness, specularBlinnPhongRoughness (default on mobile)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_COLOR in GlslViewer is u_lightColor
    - CAMERA_POSITION: in GlslViewer is u_camera
    - RAYMARCH_AO: enabled raymarched ambient occlusion
examples:
    - /shaders/lighting_raymarching_pbr.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define CAMERA_POSITION vec3(0.0, 0.0, -10.0)

fn pbr(mat: Material, shadingData: ShadingData) -> vec4f {
    shadingDataNew(mat, shadingData);

    // Indirect Lights ( Image Based Lighting )
    // ----------------------------------------
    
    lightIBLEvaluate(mat, shadingData);

    // Direct Lights
    // -------------

    {
        LightDirectional L = LightDirectionalNew();
        lightResolve(L, mat, shadingData);
        LightPoint L = LightPointNew();
        lightResolve(L, mat, shadingData);

        for (int i = 0; i < LIGHT_POINTS_TOTAL; i++) {
            LightPoint L = LIGHT_POINTS[i];
            lightResolve(L, mat, shadingData);
        }
    }

    
    // Final Sum
    // ---------
    let color = vec4f(0.0, 0.0, 0.0, 1.0);

    // Diffuse
    color.rgb  += shadingData.indirectDiffuse;
    color.rgb  += shadingData.directDiffuse;

    // Specular
    color.rgb  += shadingData.indirectSpecular;
    color.rgb  += shadingData.directSpecular; 
    color.rgb  += mat.emissive;
    color.a     = mat.albedo.a;

    return color;
}

fn pbra(mat: Material) -> vec4f {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - mat.position);
    return pbr(mat, shadingData);
}
