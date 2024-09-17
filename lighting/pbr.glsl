#ifndef DIFFUSE_FNC
#define DIFFUSE_FNC diffuseLambertConstant
#endif

#ifndef SPECULAR_FNC
#define SPECULAR_FNC specularCookTorrance
#endif

#include "../math/saturate.glsl"
#include "shadingData/new.glsl"
#include "material.glsl"
#include "light/new.glsl"
#include "light/resolve.glsl"
#include "light/iblEvaluate.glsl"

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

#ifndef CAMERA_POSITION
#define CAMERA_POSITION vec3(0.0, 0.0, -10.0)
#endif

#ifndef FNC_PBR
#define FNC_PBR

vec4 pbr(const Material mat, ShadingData shadingData) {
    shadingDataNew(mat, shadingData);

    // Indirect Lights ( Image Based Lighting )
    // ----------------------------------------
    
    lightIBLEvaluate(mat, shadingData);

    // Direct Lights
    // -------------

    {
        #if defined(LIGHT_DIRECTION)
        LightDirectional L = LightDirectionalNew();
        lightResolve(L, mat, shadingData);
        #elif defined(LIGHT_POSITION)
        LightPoint L = LightPointNew();
        lightResolve(L, mat, shadingData);
        #endif

        #if defined(LIGHT_POINTS) && defined(LIGHT_POINTS_TOTAL)
        for (int i = 0; i < LIGHT_POINTS_TOTAL; i++) {
            LightPoint L = LIGHT_POINTS[i];
            lightResolve(L, mat, shadingData);
        }
        #endif
    }

    
    // Final Sum
    // ---------
    vec4 color  = vec4(0.0, 0.0, 0.0, 1.0);

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

vec4 pbr(const in Material mat) {
    ShadingData shadingData = shadingDataNew();
    shadingData.V = normalize(CAMERA_POSITION - mat.position);
    return pbr(mat, shadingData);
}

#endif
