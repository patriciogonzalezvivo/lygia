#include "../math/const.hlsl"
#include "../space/rotate.hlsl"
#include "../space/lookAtView.hlsl"
#include "raymarch/render.hlsl"
#include "material/zero.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Raymarching template where it needs to define a float4 raymarchMSap( in float3 pos )
use: 
    - <float4> raymarch(<float4x4> viewMatrix, <float2> st)
    - <float4> raymarch(<float4x4> viewMatrix, <float2> st, out <float> eyeDepth)
    - <float4> raymarch(<float4x4> viewMatrix, <float2> st, out <float> eyeDepth, out <Material> mat)
    - <float4> raymarch(<float4x4> viewMatrix, <float2> st, out <float3> eyeDepth, out <float3> worldPosition, out <float3> worldNormal)
    - <float4> raymarch(<float3> cameraPosition, <float3> lookAt, <float2> st)
    - <float4> raymarch(<float3> cameraPosition, <float3> lookAt, <float2> st, out <float> eyeDepth)
    - <float4> raymarch(<float3> cameraPosition, <float3> lookAt, <float2> st, out <float> eyeDepth, out <Material> mat)
    - <float4> raymarch(<float3> cameraPosition, <float3> lookAt, <float2> st, out <float3> eyeDepth, out <float3> worldPosition, out <float3> worldNormal)
options:
    - RAYMARCH_RENDER_FNC(RO, RD): default raymarchDefaultRender(RO, RD, TA)
    - RAYMARCH_CAMERA_FOV: Filed of view express in degrees. Default 60.0 degrees
    - RAYMARCH_MULTISAMPLE: default 1. If it is greater than 1 it will render multisample
    - RAYMARCH_RETURN:  0. nothing (default), 1. depth;  2. depth and material
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RAYMARCH_RENDER_FNC
#define RAYMARCH_RENDER_FNC raymarchDefaultRender
#endif

#ifndef RESOLUTION
#define RESOLUTION _ScreenParams.xy
#endif

#ifndef RAYMARCH_CAMERA_FOV
#define RAYMARCH_CAMERA_FOV 60.0
#endif

#ifndef ENG_RAYMARCH
#define ENG_RAYMARCH

#if defined(RAYMARCH_MULTISAMPLE)
const float RAYMARCH_MULTISAMPLE_FACTOR = 1.0/float(RAYMARCH_MULTISAMPLE);
#endif

float4 raymarch(float4x4 viewMatrix, float2 st,
#if RAYMARCH_RETURN >= 1
                ,out float eyeDepth
#endif
#if RAYMARCH_RETURN == 2
                ,out Material mat
#endif
    ) {

    float fov = 1.0 / tan(RAYMARCH_CAMERA_FOV * DEG2RAD * 0.5);
    float3 cameraPosition = float3(viewMatrix._m03, viewMatrix._m13, viewMatrix._m23);
    float3 cameraForward = float3(viewMatrix._m02, viewMatrix._m12, viewMatrix._m22);

#if defined(RAYMARCH_MULTISAMPLE)
    float4 color = float4(0.0, 0.0, 0.0, 0.0);

    #if RAYMARCH_RETURN >= 1
    eyeDepth = 0.0;
    #endif
    #if RAYMARCH_RETURN == 2
    Material tmp;
    materialZero(tmp);
    #endif

    float2 pixel = 1.0/RESOLUTION;
    float2 offset = rotate( float2(0.5, 0.0), EIGHTH_PI);

    for (int i = 0; i < RAYMARCH_MULTISAMPLE; i++) {
        float3 rayDirection = mul((float3x3)viewMatrix, normalize(float3((st + offset * pixel)*2.0-1.0, fov)));
        
        #if RAYMARCH_RETURN >= 1
        float sampleDepth = 0.0;
        #endif

        #if RAYMARCH_RETURN == 0
            color += RAYMARCH_RENDER_FNC(camera, rayDirection, cameraForward);

        #elif RAYMARCH_RETURN == 1
            color += RAYMARCH_RENDER_FNC(camera, rayDirection, cameraForward, sampleDepth);
        
        #elif RAYMARCH_RETURN == 2
            color += RAYMARCH_RENDER_FNC(camera, rayDirection, cameraForward, sampleDepth, mat);
        #endif
        
        #if RAYMARCH_RETURN >= 1
            eyeDepth += sampleDepth;
        #endif

        #if RAYMARCH_RETURN == 2
            // Accumulate material properties
            tmp.albedo += mat.albedo;
            tmp.emissive += mat.emissive;
            tmp.position += mat.position;
            tmp.normal += mat.normal;

            #if defined(SCENE_BACK_SURFACE)
            tmp.normal_back += mat.normal_back;
            #endif

            tmp.ior += mat.ior;
            tmp.f0 += mat.f0;
            tmp.roughness += mat.roughness;
            tmp.metallic += mat.metallic;
            tmp.ambientOcclusion += mat.ambientOcclusion;

            #if defined(SHADING_MODEL_CLEAR_COAT)
            tmp.clearCoat += mat.clearCoat;
            tmp.clearCoatRoughness += mat.clearCoatRoughness;
            #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
            tmp.clearCoatNormal += mat.clearCoatNormal;
            #endif
            #endif

            #if defined(SHADING_MODEL_IRIDESCENCE)
            tmp.thickness  += mat.thickness;
            #endif

            #if defined(SHADING_MODEL_SUBSURFACE)
            tmp.subsurfaceColor += mat.subsurfaceColor;
            tmp.subsurfacePower += mat.subsurfacePower;
            tmp.subsurfaceThickness += mat.subsurfaceThickness;
            #endif

            #if defined(SHADING_MODEL_CLOTH)
            tmp.sheenColor += mat.sheenColor;
            #endif

            #if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
            tmp.specularColor += mat.specularColor;
            tmp.glossiness += mat.glossiness;
            #endif

            // I don't thing nobody needs this
            // tmp.V += mat.V;
            // tmp.R += mat.R;
            // tmp.NoV += mat.NoV;
        #endif

        offset = rotate(offset, HALF_PI);
    }

    #if RAYMARCH_RETURN >= 1
        eyeDepth *= RAYMARCH_MULTISAMPLE_FACTOR;
    #endif

    #if RAYMARCH_RETURN == 2

        // Average material
        mat.albedo = tmp.albedo * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.emissive = tmp.emissive * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.position = tmp.position * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.normal = tmp.normal * RAYMARCH_MULTISAMPLE_FACTOR;
        #if defined(SCENE_BACK_SURFACE)
        mat.normal_back = tmp.normal_back * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        mat.ior = tmp.ior * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.f0 = tmp.f0 * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.roughness = tmp.roughness * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.metallic = tmp.metallic * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.ambientOcclusion = tmp.ambientOcclusion * RAYMARCH_MULTISAMPLE_FACTOR;
        #if defined(SHADING_MODEL_CLEAR_COAT)
        mat.clearCoat = tmp.clearCoat * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.clearCoatRoughness = tmp.clearCoatRoughness * RAYMARCH_MULTISAMPLE_FACTOR;
        #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
        mat.clearCoatNormal = tmp.clearCoatNormal * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        #endif
        #if defined(SHADING_MODEL_IRIDESCENCE)
        mat.thickness = tmp.thickness * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        #if defined(SHADING_MODEL_SUBSURFACE)
        mat.subsurfaceColor = tmp.subsurfaceColor * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.subsurfacePower = tmp.subsurfacePower * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.subsurfaceThickness = tmp.subsurfaceThickness * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        #if defined(SHADING_MODEL_CLOTH)
        mat.sheenColor = tmp.sheenColor * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        #if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
        mat.specularColor = tmp.specularColor * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.glossiness = tmp.glossiness * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        // I don't thing nobody needs this
        // mat.V = tmp.V * RAYMARCH_MULTISAMPLE_FACTOR;
        // mat.R = tmp.R * RAYMARCH_MULTISAMPLE_FACTOR;
        // mat.NoV = tmp.NoV * RAYMARCH_MULTISAMPLE_FACTOR;
    #endif
    
    return color * RAYMARCH_MULTISAMPLE_FACTOR;
#else
    float3 rayDirection = mul((float3x3)viewMatrix, normalize(float3(st * 2.0 - 1.0, fov)));
    
    return RAYMARCH_RENDER_FNC( camera, rayDirection, cameraForward
    #if RAYMARCH_RETURN >= 1
                                ,eyeDepth
    #endif
    #if RAYMARCH_RETURN == 2
                                ,mat
    #endif
                                );
#endif
}

float4 raymarch(float3 cameraPosition, float3 cameraLookAt, float2 st
#if RAYMARCH_RETURN >= 1
                ,out float depth
#endif
#if RAYMARCH_RETURN == 2
                ,out Material mat
#endif
    ){
    float4x4 viewMatrix = lookAtView(cameraPosition, cameraLookAt);

    return raymarch(viewMatrix, st, 
    #if RAYMARCH_RETURN >= 1
                    ,depth
    #endif
    #if RAYMARCH_RETURN == 2
                    ,mat
    #endif
    );
}

#endif
