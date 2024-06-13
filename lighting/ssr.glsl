
#include "../space/view2screenPosition.glsl"
#include "../sdf/lineSDF.glsl"
#include "../sdf/planeSDF.glsl"
#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: ScreenSpace Reflections
use: <float> ssao(<SAMPLER_TYPE> texPosition, <SAMPLER_TYPE> texNormal, vec2 <st> [, <float> radius, float <bias>])
options:
    - SSR_MAX_STEP: number max number of raymarching steps (int)
    - SSR_MAX_DISTANCE: max distance (float)
    - CAMERA_PROJECTION_MATRIX: camera projection mat4 matrix
    - CAMERA_NEAR_CLIP: camera near clip distance
    - CAMERA_FAR_CLIP: camera far clip distance
    - SSR_FRESNEL: if define scale the opacity based on the fresnel angle
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define SSR_FRESNEL

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

vec2 ssr(SAMPLER_TYPE texPosition, SAMPLER_TYPE texNormal, vec2 st, vec2 pixel, inout float op, inout float dist) {
    vec3 viewPosition = SAMPLER_FNC(texPosition, st).xyz;
    // if (-viewPosition.z >= CAMERA_FAR_CLIP)
    //     return st;

    vec2 ssr_uv = st;
    float thickness = 0.05;
    float opacity = op;
    op = 0.0;
    dist = 0.0;

    vec3 viewNormal = SAMPLER_FNC(texNormal, st).xyz;

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

        vec3 vP = SAMPLER_FNC(texPosition, uv).xyz;
        // if (-vP.z >= CAMERA_FAR_CLIP) 
            // continue;

        // https://comp.nus.edu.sg/~lowkl/publications/lowk_persp_interp_techrep.pdf
        float recipVPZ = 1.0/viewPosition.z;
        float s = length(xy-d0)/totalLen;
        float viewReflectRayZ = 1.0 / (recipVPZ + s * (1.0/d1viewPosition.z - recipVPZ) );

        if (viewReflectRayZ <= vP.z) {
            if (lineSDF(vP, viewPosition, d1viewPosition) <= max(0.0, thickness)) {
                vec3 vN = SAMPLER_FNC(texNormal, uv).xyz;

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

vec2 ssr(SAMPLER_TYPE texPosition, SAMPLER_TYPE texNormal, vec2 st, vec2 pixel, inout float op) {
    float dist = 0.0;
    return ssr(texPosition, texNormal, st, pixel, op, dist);
}

vec3 ssr(SAMPLER_TYPE tex, SAMPLER_TYPE texPosition, SAMPLER_TYPE texNormal, vec2 st, vec2 pixel, float opacity) {
    vec3 color = SAMPLER_FNC(tex, st).rgb;
    float dist = 0.0;
    vec2 uv = ssr(texPosition, texNormal, st, pixel, opacity, dist);
    return mix(color, SAMPLER_FNC(tex, uv).rgb, opacity);
}

vec3 ssr(SAMPLER_TYPE tex, SAMPLER_TYPE texPosition, SAMPLER_TYPE texNormal, vec2 st, vec2 pixel) {
    return ssr(tex, texPosition, texNormal, st, pixel, 0.5); 
}

#endif