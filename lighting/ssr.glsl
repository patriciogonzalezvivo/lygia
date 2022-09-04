
#include "../space/view2screenPosition.glsl"
#include "../sdf/lineSDF.glsl"
#include "../sdf/planeSDF.glsl"

/*
author: Patricio Gonzalez Vivo
description: ScreenSpace Reflections
use: <float> ssao(<sampler2D> texPosition, <sampler2D> texNormal, vec2 <st> [, <float> radius, float <bias>])
options:
    - SSR_MAX_STEP: number max number of raymarching steps (int)
    - SSR_MAX_DISTANCE: max distance (float)
    - PROJECTION_MATRIX: camera projection mat4 matrix
    - RESOLUTION_SCREEN: vec2 with screen resolution
    - CAMERA_NEAR_CLIP: camera near clip distance
    - CAMERA_FAR_CLIP: camera far clip distance
    - SSR_FRESNEL: if define scale the opacity based on the fresnel angle 

license: |
    Copyright (c) 2022 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

// #define SSR_FRESNEL

#ifndef SAMPLE_FNC
#define SAMPLE_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef CAMERA_NEAR_CLIP
#define CAMERA_NEAR_CLIP u_cameraNearClip
#endif

#ifndef CAMERA_FAR_CLIP
#define CAMERA_FAR_CLIP u_cameraFarClip
#endif

#ifndef SSR_MAX_STEP
#define SSR_MAX_STEP 500
#endif

#ifndef SSR_MAX_DISTANCE
#define SSR_MAX_DISTANCE 180.0
#endif

#ifndef FNC_SSR
#define FNC_SSR

vec2 ssr(sampler2D texPosition, sampler2D texNormal, vec2 st, vec2 pixel, inout float op, inout float dist) {
    vec3 viewPosition = SAMPLE_FNC(texPosition, st).xyz;
    // if (-viewPosition.z >= CAMERA_FAR_CLIP)
        // return st;

    vec2 ssr_uv = st;
    float thickness = 0.05;
    float opacity = op;
    op = 0.0;
    dist = 0.0;

    vec3 viewNormal = SAMPLE_FNC(texNormal, st).xyz;

    vec3 viewIncidentDir = normalize(viewPosition);
    vec3 viewReflectDir = reflect(viewIncidentDir, viewNormal);
    float maxReflectRayLen = SSR_MAX_DISTANCE / dot(-viewIncidentDir, viewNormal);

    vec3 d1viewPosition = viewPosition + viewReflectDir * maxReflectRayLen;
    if (d1viewPosition.z > -CAMERA_NEAR_CLIP) {
        //https://tutorial.math.lamar.edu/Classes/CalcIII/EqnsOfLines.aspx
        float t = (-CAMERA_NEAR_CLIP - viewPosition.z)/viewReflectDir.z;
        d1viewPosition = viewPosition + viewReflectDir * t;
    }
    vec2 resolution = 1.0 / pixel;
    vec2 d0 = st * resolution;
    vec2 d1 = view2screenPosition(d1viewPosition) * resolution;

    float totalLen = length(d1-d0);
    float xLen = d1.x-d0.x;
    float yLen = d1.y-d0.y;
    float totalStep = max(abs(xLen),abs(yLen));
    float xSpan = (xLen/totalStep);// * pixel.x;
    float ySpan = (yLen/totalStep);// * pixel.y;

    for (float i = 0.0; i < float(SSR_MAX_STEP); i++) {
        if ( i >= totalStep) 
            break;

        vec2 xy = vec2(d0.x + i*xSpan, d0.y + i*ySpan);
        vec2 uv = xy * pixel;

        vec3 vP = SAMPLE_FNC(texPosition, uv).xyz;
        // if (-vP.z >= CAMERA_FAR_CLIP) 
        //     continue;

        // https://comp.nus.edu.sg/~lowkl/publications/lowk_persp_interp_techrep.pdf
        float recipVPZ = 1.0/viewPosition.z;
        float s = length(xy-d0)/totalLen;
        float viewReflectRayZ = 1.0 / (recipVPZ + s * (1.0/d1viewPosition.z - recipVPZ) );

        if (viewReflectRayZ <= vP.z) {
            if (lineSDF(vP, viewPosition, d1viewPosition) <= max(0.0, thickness)) {
                vec3 vN = SAMPLE_FNC(texNormal, uv).xyz;

                if (dot(viewReflectDir, vN) >= 0.0) 
                    continue;

                dist = planeSDF(vP, viewPosition, viewNormal);
                if (dist > SSR_MAX_DISTANCE) 
                    break;

                dist = s;

                op = opacity;
                #ifdef SSR_FRESNEL
                    float fresnelCoe = (dot(viewIncidentDir, viewReflectDir) + 1.0) * 0.5;
                    op *= fresnelCoe;
                #endif

                ssr_uv = uv;
                break;
            }
        }
    }

    return ssr_uv;
}

vec2 ssr(sampler2D texPosition, sampler2D texNormal, vec2 st, vec2 pixel, inout float op) {
    float dist = 0.0;
    return ssr(texPosition, texNormal, st, pixel, op, dist);
}

vec3 ssr(sampler2D tex, sampler2D texPosition, sampler2D texNormal, vec2 st, vec2 pixel, float opacity) {
    vec3 color = SAMPLE_FNC(tex, st).rgb;
    float dist = 0.0;
    vec2 uv = ssr(texPosition, texNormal, st, pixel, opacity, dist);
    return mix(color, SAMPLE_FNC(tex, uv).rgb, opacity);
}

vec3 ssr(sampler2D tex, sampler2D texPosition, sampler2D texNormal, vec2 st, vec2 pixel) {
    return ssr(tex, texPosition, texNormal, st, pixel, 0.5); 
}

#endif