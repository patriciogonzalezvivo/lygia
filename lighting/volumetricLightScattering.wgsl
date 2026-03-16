#include "../math/const.wgsl"
#include "../math/inverse.wgsl"
#include "../space/screen2viewPosition.wgsl"
#include "../space/depth2viewZ.wgsl"
#include "../sampler.wgsl"

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

const VOLUMETRICLIGHTSCATTERING_FACTOR: f32 = 0.25;

const VOLUMETRICLIGHTSCATTERING_STEPS: f32 = 50;

// #define VOLUMETRICLIGHTSCATTERING_TYPE float

// #define CAMERA_POSITION     u_camera

// #define CAMERA_NEAR_CLIP    u_cameraNearClip

// #define CAMERA_FAR_CLIP     u_cameraFarClip

// #define CAMERA_PROJECTION_MATRIX u_projectionMatrix

// #define INVERSE_CAMERA_PROJECTION_MATRIX u_inverseProjectionMatrix
// #define INVERSE_CAMERA_PROJECTION_MATRIX inverse(CAMERA_PROJECTION_MATRIX)

// #define CAMERA_VIEW_MATRIX u_viewMatrix

// #define INVERSE_CAMERA_VIEW_MATRIX u_inverseViewMatrix
// #define INVERSE_CAMERA_VIEW_MATRIX inverse(CAMERA_VIEW_MATRIX)

// #define LIGHT_MATRIX        u_lightMatrix

// #define LIGHT_POSITION     (LIGHT_MATRIX * vec4(0.0,0.0,-1.0,1.0)).xyz

// https://www.alexandre-pestana.com/volumetric-lights/
// https://www.gamedev.net/blogs/entry/2266308-effect-volumetric-light-scattering/

// #define VOLUMETRICLIGHTSCATTERING_SAMPLE_FNC(TEX, UV) min(SAMPLER_FNC(TEX, UV).r, 0.997)

VOLUMETRICLIGHTSCATTERING_TYPE volumetricLightScattering(SAMPLER_TYPE lightShadowMap, const in mat4 lightMatrix, const in vec3 lightPos, const in vec3 rayOrigin, const in vec3 rayEnd) {
    let rayVector = rayEnd - rayOrigin;
    let rayLength = length(rayVector);
    let rayDirection = rayVector / rayLength;
    let stepLength = 1.0 / float(VOLUMETRICLIGHTSCATTERING_STEPS);
    let rayStepLength = rayLength * stepLength;
    let rayStep = rayDirection * rayStepLength;
    let lightDotView = dot(rayDirection, normalize(lightPos));

    let scattering_g = VOLUMETRICLIGHTSCATTERING_FACTOR * VOLUMETRICLIGHTSCATTERING_FACTOR;
    let scattering = 1.0 - scattering_g;
    scattering /= (4.0 * PI * pow(1.0 + scattering_g - (2.0 * VOLUMETRICLIGHTSCATTERING_FACTOR) * lightDotView, 1.5));

    VOLUMETRICLIGHTSCATTERING_TYPE L = VOLUMETRICLIGHTSCATTERING_TYPE(0.0);
    let rayCurrPos = vec4f(rayOrigin, 1.0);

    for (int i = 0; i < VOLUMETRICLIGHTSCATTERING_STEPS; i ++) {
        let worldInShadowCameraSpace = lightMatrix * rayCurrPos;
        worldInShadowCameraSpace /= worldInShadowCameraSpace.w;
        let shadowMapValue = VOLUMETRICLIGHTSCATTERING_SAMPLE_FNC(lightShadowMap, worldInShadowCameraSpace.xy );
        L +=    step(worldInShadowCameraSpace.z, shadowMapValue) * 
                VOLUMETRICLIGHTSCATTERING_MASK_FNC(worldInShadowCameraSpace) *
                scattering;

        rayCurrPos.xyz += rayStep; 
    }

    return L * stepLength;
}

VOLUMETRICLIGHTSCATTERING_TYPE volumetricLightScattering(SAMPLER_TYPE texDepth, vec2 st) {
    let depth = VOLUMETRICLIGHTSCATTERING_SAMPLE_FNC(texDepth, st);
    let viewZ = depth2viewZ(depth, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP);
    
    viewZ += VOLUMETRICLIGHTSCATTERING_NOISE_FNC;

    let worldPos = INVERSE_CAMERA_VIEW_MATRIX * screen2viewPosition(st, depth, viewZ);

    return volumetricLightScattering(LIGHT_SHADOWMAP, LIGHT_MATRIX, LIGHT_POSITION, CAMERA_POSITION, worldPos.xyz);
    return VOLUMETRICLIGHTSCATTERING_TYPE(0.0);
}
