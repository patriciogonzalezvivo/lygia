#include "../math/const.glsl"
#include "../math/inverse.glsl"
#include "../space/screen2viewPosition.glsl"
#include "../space/depth2viewZ.glsl"
#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: ScreenSpace Reflections
use: <float> ssao(<SAMPLER_TYPE> texPosition, <SAMPLER_TYPE> texNormal, vec2 <st> [, <float> radius, float <bias>])
options:
    - CAMERA_POSITION
    - CAMERA_NEAR_CLIP: camera near clip distance
    - CAMERA_FAR_CLIP: camera far clip distance
    - CAMERA_ORTHOGRAPHIC_PROJECTION, if it's not present is consider a PERECPECTIVE camera
    - CAMERA_PROJECTION_MATRIX: mat4 matrix with camera projection
    - CAMERA_VIEW_MATRIX
    - INVERSE_CAMERA_VIEW_MATRIX
    - INVERSE_CAMERA_PROJECTION_MATRIX: mat4 matrix with the inverse camara projection
    - LIGHT_POSITION (optional)
    - LIGHT_MATRIX
    - LIGHT_SHADOWMAP
    - VOLUMETRICLIGHTSCATTERING_FACTOR
    - VOLUMETRICLIGHTSCATTERING_STEPS
    - VOLUMETRICLIGHTSCATTERING_NOISE_FNC
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef VOLUMETRICLIGHTSCATTERING_FACTOR
#define VOLUMETRICLIGHTSCATTERING_FACTOR 0.25
#endif

#ifndef VOLUMETRICLIGHTSCATTERING_STEPS
#define VOLUMETRICLIGHTSCATTERING_STEPS 50
#endif

#ifndef VOLUMETRICLIGHTSCATTERING_TYPE
#define VOLUMETRICLIGHTSCATTERING_TYPE float
#endif

#ifndef CAMERA_POSITION
#define CAMERA_POSITION     u_camera
#endif

#ifndef CAMERA_NEAR_CLIP
#define CAMERA_NEAR_CLIP    u_cameraNearClip
#endif

#ifndef CAMERA_FAR_CLIP
#define CAMERA_FAR_CLIP     u_cameraFarClip
#endif

#ifndef CAMERA_PROJECTION_MATRIX
#define CAMERA_PROJECTION_MATRIX u_projectionMatrix
#endif

#ifndef INVERSE_CAMERA_PROJECTION_MATRIX
// #define INVERSE_CAMERA_PROJECTION_MATRIX u_inverseProjectionMatrix
#define INVERSE_CAMERA_PROJECTION_MATRIX inverse(CAMERA_PROJECTION_MATRIX)
#endif

#ifndef CAMERA_VIEW_MATRIX
#define CAMERA_VIEW_MATRIX u_viewMatrix
#endif

#ifndef INVERSE_CAMERA_VIEW_MATRIX
// #define INVERSE_CAMERA_VIEW_MATRIX u_inverseViewMatrix
#define INVERSE_CAMERA_VIEW_MATRIX inverse(CAMERA_VIEW_MATRIX)
#endif

#ifndef LIGHT_MATRIX
#define LIGHT_MATRIX        u_lightMatrix
#endif

#ifndef LIGHT_POSITION
#define LIGHT_POSITION     (LIGHT_MATRIX * vec4(0.0,0.0,-1.0,1.0)).xyz
#endif

// https://www.alexandre-pestana.com/volumetric-lights/
// https://www.gamedev.net/blogs/entry/2266308-effect-volumetric-light-scattering/

#ifndef VOLUMETRICLIGHTSCATTERING_SAMPLE_FNC
#define VOLUMETRICLIGHTSCATTERING_SAMPLE_FNC(TEX, UV) min(SAMPLER_FNC(TEX, UV).r, 0.997)
#endif

#ifndef FNC_VOLUMETRICLIGHTSCATTERING
#define FNC_VOLUMETRICLIGHTSCATTERING
VOLUMETRICLIGHTSCATTERING_TYPE volumetricLightScattering(SAMPLER_TYPE lightShadowMap, const in mat4 lightMatrix, const in vec3 lightPos, const in vec3 rayOrigin, const in vec3 rayEnd) {
    vec3  rayVector     = rayEnd - rayOrigin;
    float rayLength     = length(rayVector);
    vec3  rayDirection  = rayVector / rayLength;
    float stepLength    = 1.0 / float(VOLUMETRICLIGHTSCATTERING_STEPS);
    float rayStepLength = rayLength * stepLength;
    vec3  rayStep       = rayDirection * rayStepLength;
    float lightDotView  = dot(rayDirection, normalize(lightPos));

    float scattering_g  = VOLUMETRICLIGHTSCATTERING_FACTOR * VOLUMETRICLIGHTSCATTERING_FACTOR;
    float scattering    = 1.0 - scattering_g;
    scattering /= (4.0 * PI * pow(1.0 + scattering_g - (2.0 * VOLUMETRICLIGHTSCATTERING_FACTOR) * lightDotView, 1.5));

    VOLUMETRICLIGHTSCATTERING_TYPE L = VOLUMETRICLIGHTSCATTERING_TYPE(0.0);
    vec4  rayCurrPos    = vec4(rayOrigin, 1.0);

    for (int i = 0; i < VOLUMETRICLIGHTSCATTERING_STEPS; i ++) {
        vec4 worldInShadowCameraSpace = lightMatrix * rayCurrPos;
        worldInShadowCameraSpace /= worldInShadowCameraSpace.w;
        float shadowMapValue = VOLUMETRICLIGHTSCATTERING_SAMPLE_FNC(lightShadowMap, worldInShadowCameraSpace.xy );
        L +=    step(worldInShadowCameraSpace.z, shadowMapValue) * 
                #ifdef VOLUMETRICLIGHTSCATTERING_MASK_FNC
                VOLUMETRICLIGHTSCATTERING_MASK_FNC(worldInShadowCameraSpace) *
                #endif
                scattering;

        rayCurrPos.xyz += rayStep; 
    }

    return L * stepLength;
}

VOLUMETRICLIGHTSCATTERING_TYPE volumetricLightScattering(SAMPLER_TYPE texDepth, vec2 st) {
    float depth = VOLUMETRICLIGHTSCATTERING_SAMPLE_FNC(texDepth, st);
    float viewZ = depth2viewZ(depth, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP);
    
    #ifdef VOLUMETRICLIGHTSCATTERING_NOISE_FNC
    viewZ += VOLUMETRICLIGHTSCATTERING_NOISE_FNC;
    #endif

    vec4 worldPos = INVERSE_CAMERA_VIEW_MATRIX * screen2viewPosition(st, depth, viewZ);

    #ifdef LIGHT_SHADOWMAP
    return volumetricLightScattering(LIGHT_SHADOWMAP, LIGHT_MATRIX, LIGHT_POSITION, CAMERA_POSITION, worldPos.xyz);
    #else 
    return VOLUMETRICLIGHTSCATTERING_TYPE(0.0);
    #endif
}

#endif