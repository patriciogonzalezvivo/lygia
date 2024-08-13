/*
contributors: Patricio Gonzalez Vivo
description: Raymarching template where it needs to define a vec4 raymarchMap( in vec3 pos )
use: <vec4> raymarch(<vec3> cameraPosition, <vec3> lookAt, <vec2> st,
    out <vec3> eyeDepth, out <vec3> worldPosition, out <vec3> worldNormal )
options:
    - RAYMARCH_CAMERA_MATRIX_FNC(RO, TA): default raymarchCamera(RO, TA)
    - RAYMARCH_RENDER_FNC(RO, RD): default raymarchDefaultRender(RO, RD, TA)
    - RAYMARCH_CAMERA_FOV
examples:
    - /shaders/lighting_raymarching.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RAYMARCH_RENDER_FNC
#define RAYMARCH_RENDER_FNC raymarchDefaultRender
#endif

#ifndef RAYMARCH_CAMERA_FOV
#define RAYMARCH_CAMERA_FOV 60.0
#endif

#include "../math/const.glsl"
#include "../space/rotate.glsl"
#include "../space/lookAtView.glsl"
#include "raymarch/render.glsl"

#ifndef ENG_RAYMARCHING
#define ENG_RAYMARCHING

#if defined(RAYMARCH_MULTISAMPLE)
const float RAYMARCH_MULTISAMPLE_FACTOR = 1.0/float(RAYMARCH_MULTISAMPLE);
#endif

vec4 raymarch( mat4 viewMatrix, vec2 st,
    out float eyeDepth, out vec3 worldPos, out vec3 worldNormal) {

    float fov = 1.0/tan(RAYMARCH_CAMERA_FOV*DEG2RAD/2.0);
    vec3 camera = vec3(viewMatrix[0][3], viewMatrix[1][3], viewMatrix[2][3]);
    vec3 cameraForward = vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]);

#if defined(RAYMARCH_MULTISAMPLE)
    vec4 color = vec4(0.0);
    eyeDepth = 0.0;
    worldPos = vec3(0.0);
    worldNormal = vec3(0.0);
    vec2 pixel = 1.0/RESOLUTION;
    vec2 offset = rotate( vec2(0.5, 0.0), HALF_PI/4.);

    for (int i = 0; i < RAYMARCH_MULTISAMPLE; i++) {
        vec3 rd = mat3(viewMatrix) * normalize(vec3((st + offset * pixel)*2.0-1.0, fov));
        float sampleDepth = 0.0;
        vec3 sampleWorldPos = vec3(0.0);
        vec3 sampleWorldNormal = vec3(0.0);
        color += RAYMARCH_RENDER_FNC( camera, rd, cameraForward, sampleDepth, sampleWorldPos, sampleWorldNormal );
        eyeDepth += sampleDepth;
        worldPos += sampleWorldPos;
        worldNormal += sampleWorldNormal;
        offset = rotate(offset, HALF_PI);
    }
    eyeDepth *= RAYMARCH_MULTISAMPLE_FACTOR;
    worldPos *= RAYMARCH_MULTISAMPLE_FACTOR;
    worldNormal *= RAYMARCH_MULTISAMPLE_FACTOR;
    return color * RAYMARCH_MULTISAMPLE_FACTOR;
#else
    vec3 rd = mat3(viewMatrix) * normalize(vec3((st + offset * pixel)*2.0-1.0, fov));
    return RAYMARCH_RENDER_FNC( camera, rd, cameraForward, eyeDepth, worldPos, worldNormal );
#endif
}

/*vec4 raymarch( mat4 viewMatrix, vec2 st, out Material mat, out float eyeDepth) {
    float fov = 1.0/tan(RAYMARCH_CAMERA_FOV*DEG2RAD/2.0);
    vec3 camera = vec3(viewMatrix[0][3], viewMatrix[1][3], viewMatrix[2][3]);
    vec3 cameraForward = vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]);

#if defined(RAYMARCH_MULTISAMPLE)
    vec4 color = vec4(0.0);
    eyeDepth = 0.0;

    Material tmp;
    materialZero(tmp);

    vec3 worldPos = vec3(0.0);
    vec3 worldNormal = vec3(0.0);
    vec2 pixel = 1.0/RESOLUTION;
    vec2 offset = rotate( vec2(0.5, 0.0), HALF_PI/4.);

    for (int i = 0; i < RAYMARCH_MULTISAMPLE; i++) {
        vec3 rd = mat3(viewMatrix) * normalize(vec3((st + offset * pixel)*2.0-1.0, fov));
        float sampleDepth = 0.0;
        vec3 sampleWorldPos = vec3(0.0);
        vec3 sampleWorldNormal = vec3(0.0);
        color += RAYMARCH_RENDER_FNC( camera, rd, cameraForward, mat, sampleDepth, sampleWorldPos, sampleWorldNormal);
        eyeDepth += sampleDepth;
        
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

        // tmp.V += mat.V;
        // tmp.R += mat.R;
        // tmp.NoV += mat.NoV;

        offset = rotate(offset, HALF_PI);
    }
    eyeDepth *= RAYMARCH_MULTISAMPLE_FACTOR;

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
    // mat.V = tmp.V * RAYMARCH_MULTISAMPLE_FACTOR;
    // mat.R = tmp.R * RAYMARCH_MULTISAMPLE_FACTOR;
    // mat.NoV = tmp.NoV * RAYMARCH_MULTISAMPLE_FACTOR;
    
    return color * RAYMARCH_MULTISAMPLE_FACTOR;
#else
    vec3 rd = mat3(viewMatrix) * normalize(vec3((st + offset * pixel)*2.0-1.0, fov));
    vec3 sampleWorldPos = vec3(0.0);
    vec3 sampleWorldNormal = vec3(0.0);
    return RAYMARCH_RENDER_FNC( camera, rd, cameraForward, mat, eyeDepth, sampleWorldPos, sampleWorldNormal);
#endif
}*/

vec4 raymarch(vec3 cameraPosition, vec3 cameraLookAt, vec2 st)
{
    float depth = 0.0;
    vec3 worldPos = vec3(0.0, 0.0, 0.0);
    vec3 worldNormal = vec3(0.0, 0.0, 0.0);
    mat4 viewMatrix = lookAtViewMatrix(cameraPosition, cameraLookAt);
    return raymarch(viewMatrix, st, depth, worldPos, worldNormal);
}

#endif