#include "../math/toMat3.glsl"
#include "../math/const.glsl"
#include "../space/rotate.glsl"
#include "../space/lookAtView.glsl"
#include "raymarch/render.glsl"
#include "raymarch/volume.glsl"
#include "material/zero.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Raymarching template where it needs to define a vec4 raymarchMap( in vec3 pos )
use: 
    - <vec4> raymarch(<mat4> viewMatrix, <vec2> st)
    - <vec4> raymarch(<mat4> viewMatrix, <vec2> st, out <float> eyeDepth)
    - <vec4> raymarch(<mat4> viewMatrix, <vec2> st, out <float> eyeDepth, out <Material> mat)
    - <vec4> raymarch(<mat4> viewMatrix, <vec2> st, out <vec3> eyeDepth, out <vec3> worldPosition, out <vec3> worldNormal)
    - <vec4> raymarch(<vec3> cameraPosition, <vec3> lookAt, <vec2> st)
    - <vec4> raymarch(<vec3> cameraPosition, <vec3> lookAt, <vec2> st, out <float> eyeDepth)
    - <vec4> raymarch(<vec3> cameraPosition, <vec3> lookAt, <vec2> st, out <float> eyeDepth, out <Material> mat)
    - <vec4> raymarch(<vec3> cameraPosition, <vec3> lookAt, <vec2> st, out <vec3> eyeDepth, out <vec3> worldPosition, out <vec3> worldNormal)
options:
    - RAYMARCH_RENDER_FNC(RO, RD): default raymarchDefaultRender(RO, RD, TA)
    - RAYMARCH_CAMERA_FOV: Filed of view express in degrees. Default 60.0 degrees
    - RAYMARCH_MULTISAMPLE: default 1. If it is greater than 1 it will render multisample
    - RAYMARCH_RETURN:  0. nothing (default), 1. depth;  2. depth and material
examples:
    - /shaders/lighting_raymarching.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RAYMARCH_RENDER_FNC
#define RAYMARCH_RENDER_FNC raymarchDefaultRender
#endif

#ifndef RAYMARCH_VOLUME_RENDER_FNC
#define RAYMARCH_VOLUME_RENDER_FNC raymarchVolume
#endif

#ifndef RAYMARCH_CAMERA_FOV
#define RAYMARCH_CAMERA_FOV 60.0
#endif

#ifndef ENG_RAYMARCHING
#define ENG_RAYMARCHING

#if defined(RAYMARCH_MULTISAMPLE)
const float RAYMARCH_MULTISAMPLE_FACTOR = 1.0/float(RAYMARCH_MULTISAMPLE);
#endif

vec4 raymarch(  mat4 viewMatrix, vec2 st
#if RAYMARCH_RETURN >= 1
                ,out float eyeDepth
#endif
#if RAYMARCH_RETURN == 2
                ,out Material mat
#endif
    ) {
                    
    float fov = 1.0 / tan(RAYMARCH_CAMERA_FOV * DEG2RAD * 0.5);
    vec3 camera = viewMatrix[3].xyz;
    vec3 cameraForward = viewMatrix[2].xyz;
    mat3 viewMatrix3 = toMat3(viewMatrix);

#if defined(RAYMARCH_MULTISAMPLE)
    vec4 color = vec4(0.0);

    #if RAYMARCH_RETURN >= 1
    eyeDepth = 0.0;
    #endif
    #if RAYMARCH_RETURN == 2
    Material matAcc;
    materialZero(matAcc);
    #endif

    vec2 pixel = 1.0/RESOLUTION;
    vec2 offset = rotate( vec2(0.5, 0.0), EIGHTH_PI);

    for (int i = 0; i < RAYMARCH_MULTISAMPLE; i++) {
        vec3 rayDirection = viewMatrix3 * normalize(vec3((st + offset * pixel)*2.0-1.0, fov));

        float sampleDepth = 0.0;
        float dist = 0.0;

        #if RAYMARCH_RETURN != 2
            Material mat;        
        #endif

        vec4 opaque = RAYMARCH_RENDER_FNC(camera, rayDirection, cameraForward, dist, sampleDepth, mat);
        #ifdef RAYMARCH_VOLUME
        color += vec4(RAYMARCH_VOLUME_RENDER_FNC(camera, rayDirection, st, dist, opaque.rgb), opaque.a);
        #else
        color += opaque;
        #endif

        #if RAYMARCH_RETURN >= 1
            eyeDepth += sampleDepth;
        #endif
        
        #if RAYMARCH_RETURN == 2
            // Accumulate material properties
            matAcc.albedo += mat.albedo;
            matAcc.emissive += mat.emissive;
            matAcc.position += mat.position;
            matAcc.normal += mat.normal;

            #if defined(SCENE_BACK_SURFACE)
            matAcc.normal_back += mat.normal_back;
            #endif

            matAcc.ior += mat.ior;
            matAcc.f0 += mat.f0;
            matAcc.roughness += mat.roughness;
            matAcc.metallic += mat.metallic;
            matAcc.ambientOcclusion += mat.ambientOcclusion;

            #if defined(SHADING_MODEL_CLEAR_COAT)
            matAcc.clearCoat += mat.clearCoat;
            matAcc.clearCoatRoughness += mat.clearCoatRoughness;
            #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
            matAcc.clearCoatNormal += mat.clearCoatNormal;
            #endif
            #endif

            #if defined(SHADING_MODEL_IRIDESCENCE)
            matAcc.thickness  += mat.thickness;
            #endif

            #if defined(SHADING_MODEL_SUBSURFACE)
            matAcc.subsurfaceColor += mat.subsurfaceColor;
            matAcc.subsurfacePower += mat.subsurfacePower;
            matAcc.subsurfaceThickness += mat.subsurfaceThickness;
            #endif

            #if defined(SHADING_MODEL_CLOTH)
            matAcc.sheenColor += mat.sheenColor;
            #endif

            #if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
            matAcc.specularColor += mat.specularColor;
            matAcc.glossiness += mat.glossiness;
            #endif

            // I don't thing nobody needs this
            // matAcc.V += mat.V;
            // matAcc.R += mat.R;
            // matAcc.NoV += mat.NoV;
        #endif

        offset = rotate(offset, HALF_PI);
    }

    #if RAYMARCH_RETURN >= 1
        eyeDepth *= RAYMARCH_MULTISAMPLE_FACTOR;
    #endif

    #if RAYMARCH_RETURN == 2

        // Average material
        mat.albedo = matAcc.albedo * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.emissive = matAcc.emissive * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.position = matAcc.position * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.normal = matAcc.normal * RAYMARCH_MULTISAMPLE_FACTOR;
        #if defined(SCENE_BACK_SURFACE)
        mat.normal_back = matAcc.normal_back * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        mat.ior = matAcc.ior * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.f0 = matAcc.f0 * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.roughness = matAcc.roughness * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.metallic = matAcc.metallic * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.ambientOcclusion = matAcc.ambientOcclusion * RAYMARCH_MULTISAMPLE_FACTOR;
        #if defined(SHADING_MODEL_CLEAR_COAT)
        mat.clearCoat = matAcc.clearCoat * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.clearCoatRoughness = matAcc.clearCoatRoughness * RAYMARCH_MULTISAMPLE_FACTOR;
        #if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
        mat.clearCoatNormal = matAcc.clearCoatNormal * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        #endif
        #if defined(SHADING_MODEL_IRIDESCENCE)
        mat.thickness = matAcc.thickness * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        #if defined(SHADING_MODEL_SUBSURFACE)
        mat.subsurfaceColor = matAcc.subsurfaceColor * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.subsurfacePower = matAcc.subsurfacePower * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.subsurfaceThickness = matAcc.subsurfaceThickness * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        #if defined(SHADING_MODEL_CLOTH)
        mat.sheenColor = matAcc.sheenColor * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        #if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
        mat.specularColor = matAcc.specularColor * RAYMARCH_MULTISAMPLE_FACTOR;
        mat.glossiness = matAcc.glossiness * RAYMARCH_MULTISAMPLE_FACTOR;
        #endif
        // I don't thing nobody needs this
        // mat.V = matAcc.V * RAYMARCH_MULTISAMPLE_FACTOR;
        // mat.R = matAcc.R * RAYMARCH_MULTISAMPLE_FACTOR;
        // mat.NoV = matAcc.NoV * RAYMARCH_MULTISAMPLE_FACTOR;
    #endif
    
    return color * RAYMARCH_MULTISAMPLE_FACTOR;
#else

    // Single sample
    vec3 rayDirection = viewMatrix3 * normalize(vec3(st*2.0-1.0, fov));
    float dist = 0.0;
    #if RAYMARCH_RETURN == 0
        float eyeDepth = 0.0;
    #endif
    #if RAYMARCH_RETURN != 2
        Material mat;        
    #endif

    vec4 color = RAYMARCH_RENDER_FNC( camera, rayDirection, cameraForward, dist, eyeDepth, mat);
    #ifdef RAYMARCH_VOLUME
    color.rgb = RAYMARCH_VOLUME_RENDER_FNC(camera, rayDirection, st, dist, color.rgb);
    #endif

    return color;
#endif
}

vec4 raymarch(  vec3 cameraPosition, vec3 cameraLookAt, vec2 st 
#if RAYMARCH_RETURN >= 1
                ,out float depth
#endif
#if RAYMARCH_RETURN == 2
                ,out Material mat
#endif
    ) {
    mat4 viewMatrix = lookAtView(cameraPosition, cameraLookAt);

    return raymarch(viewMatrix, st
    #if RAYMARCH_RETURN >= 1
                    ,depth
    #endif
    #if RAYMARCH_RETURN == 2
                    ,mat
    #endif
    );
}

#endif