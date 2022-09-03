#include "lygia/math/inverse.glsl"
#include "lygia/space/screen2viewPosition.glsl"
#include "lygia/space/depth2viewZ.glsl"

/*
author: Patricio Gonzalez Vivo
description: ScreenSpace Reflections
use: <float> ssao(<sampler2D> texPosition, <sampler2D> texNormal, vec2 <st> [, <float> radius, float <bias>])
options:
    - CAMERA_POSITION
    - CAMERA_NEAR_CLIP: camera near clip distance
    - CAMERA_FAR_CLIP: camera far clip distance
    - INVERSE_VIEW_MATRIX
    - INVERSE_PROJECTION_MATRIX
    - LIGHT_POSITION (optional)
    - LIGHT_MATRIX
    - LIGHT_SHADOWMAP
    - VOLUMETRICLIGHTSCATTERING_FACTOR
    - VOLUMETRICLIGHTSCATTERING_STEPS
    - VOLUMETRICLIGHTSCATTERING_NOISE_FNC

license: |
    Copyright (c) 2022 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef VOLUMETRICLIGHTSCATTERING_FACTOR
#define VOLUMETRICLIGHTSCATTERING_FACTOR 0.25
#endif

#ifndef VOLUMETRICLIGHTSCATTERING_STEPS
#define VOLUMETRICLIGHTSCATTERING_STEPS 50
#endif


#ifndef INVERSE_VIEW_MATRIX
#define INVERSE_VIEW_MATRIX inverse(u_viewMatrix)
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

#ifndef LIGHT_MATRIX
#define LIGHT_MATRIX        u_lightMatrix
#endif

#ifndef LIGHT_POSITION
#define LIGHT_POSITION     (LIGHT_MATRIX * vec4(0.0,0.0,-1.0,1.0)).xyz
#endif

// https://www.alexandre-pestana.com/volumetric-lights/
// https://www.gamedev.net/blogs/entry/2266308-effect-volumetric-light-scattering/

#ifndef FNC_VOLUMETRICLIGHTSCATTERING
#define FNC_VOLUMETRICLIGHTSCATTERING
vec3 volumetricLightScattering(sampler2D lightShadowMap, mat4 lightMatrix, vec3 lightPos, vec3 rayOrigin, vec3 rayEnd) {
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

    vec3  L             = vec3(0.0);
    vec3  rayCurrPos    = rayOrigin;

    for (int i = 0; i < VOLUMETRICLIGHTSCATTERING_STEPS; i ++) {

        vec4 worldInShadowCameraSpace = lightMatrix * vec4(rayCurrPos, 1.0);
        worldInShadowCameraSpace /= worldInShadowCameraSpace.w;
        float shadowMapValue = texture2D(lightShadowMap, worldInShadowCameraSpace.xy ).r;
        L += step(worldInShadowCameraSpace.z, shadowMapValue) * scattering;

        rayCurrPos += rayStep; 
    }

    return L * stepLength;
}

vec3 volumetricLightScattering(sampler2D texDepth, vec2 st) {
    float depth = texture2D(texDepth, st).r;
    depth = min(depth, 0.997);
    float viewZ = depth2viewZ(depth, CAMERA_NEAR_CLIP, CAMERA_FAR_CLIP);
    #ifdef VOLUMETRICLIGHTSCATTERING_NOISE_FNC
    viewZ += VOLUMETRICLIGHTSCATTERING_NOISE_FNC;
    #endif
    vec3 viewPos = screen2viewPosition(st, depth, viewZ);
    // vec3 viewPos = texture2D(u_scenePosition, st).xyz;

    vec3 worldPos = (INVERSE_VIEW_MATRIX * vec4(viewPos, 1.0)).xyz;
    // worldPos = (u_inverseViewMatrix * vec4(viewPos, 1.0)).xyz;

    #ifdef LIGHT_SHADOWMAP
    return volumetricLightScattering(LIGHT_SHADOWMAP, LIGHT_MATRIX, LIGHT_POSITION, CAMERA_POSITION, worldPos);
    #else 
    return vec3(0.0);
    #endif
}
#endif